import os

import numpy as np
import mne
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

import multiprocessing
from joblib import Parallel, delayed
from gpuparallel import GPUParallel
from gpuparallel import delayed as GPUdelayed

import model.suppfunc.Network as Network
import model.suppfunc._config as _config
import model.suppfunc._smoothing as _smoothing
import model.suppfunc._segmentation as _segmentation
import model.suppfunc._tf as _tf
import model.suppfunc._preprocessing as _preprocessing

from helperfunctions.channel_selection_helperfunctions import _get_10_20_channels

import constants as c
import globals

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

class ReshapeTransform:
    """Class to transform a PIL image with torch.reshape.

    Attributes
    ----------
    new_size : Tuple(n channels, image height, image width)

    Methods
    -------
    __call__(img): Resizes PIL image.
    """
    def __init__(self, new_size):
        """Constructs new size attribute to which a PIL image will be transformed

        Parameters
        ----------
        new_size : Tuple(n channels, image height, image width)
        """
        self.new_size = new_size

    def __call__(self, img):
        """Resizes a PIL image.

        Parameters
        ----------
        img : PIL image

        Returns
        -------
        Resized PIL image
        """
        return torch.reshape(img, self.new_size)


def _transform(imagesTF):
    """Returns converted to torch.Tensor resized PIL images.

    Args:
        imagesTF (list): List of PIL images.

    Returns:
        (torch.Tensor): Output tensor of a concatenated sequence of tensors.
    """
    trans1 = transforms.ToTensor()
    trans2 = ReshapeTransform((19, 45, 100))

    tensors = [trans2(trans1(image)) for image in imagesTF]
    return torch.stack(tensors).double()


def load_model():
    """Loads the trained CNN model.

    Returns:
        net (Network.ConvNet): Trained CNN with loaded weights.
    """
    path_to_model = os.path.join(c.MODEL_DIRECTORY, 'trained_model.pth.tar')
    # Define network
    cfg = _config.cfg
    net = Network.ConvNet(cfg).to(DEVICE)
    # Load model weights
    print(DEVICE)
    model = torch.load(path_to_model, map_location=DEVICE)
    net.load_state_dict(model['state_dict'])
    return net.double()


def _run_model(model, tf_tensor):
    """Runs the CNN model on new data and returns predicted class probabilities.

    Args:
        model (Network.ConvNet): Weights of the CNN model trained on time-frequency-transformed EEG segments.
        tf_tensor (torch.Tensor): Time-frequency-transformed EEG segments of normalized power obtained from an EEG recording and converted to torch.Tensor.

    Returns:
        probabilities (numpy.array): Predicted class probabilities.
    """
    # Reshape TF tensor (n_segments, n_channels, n_freq, n_times) to (n_segments, 1, n_channels, n_freq, n_times)
    tf_tensor = torch.reshape(tf_tensor, (tf_tensor.shape[0], 1, tf_tensor.shape[1],
                                          tf_tensor.shape[2], tf_tensor.shape[3]))
    # Apply model
    print("Applying tfCNN ...")
    print("Device {}".format(DEVICE))
    
    if DEVICE.type == 'cpu':
        num_cores = multiprocessing.cpu_count()
    elif DEVICE.type == 'cuda':
        num_cores = torch.cuda.device_count()

    def run(input):
        model.eval()
        with torch.no_grad():
            tf = input
            tf = tf.to(DEVICE)
            # Forward pass
            outputs = model(tf)
            _, predicted = torch.max(outputs, 1)
            prob = F.softmax(outputs, dim=1)
        return prob

    if DEVICE.type == 'cpu':
        probabilities = Parallel(n_jobs=num_cores, backend='threading')(delayed(run)(tf_tensor[k]) for k in range(len(tf_tensor)))
    elif DEVICE.type == 'cuda':
        probabilities = GPUParallel(n_gpu=num_cores)(GPUdelayed(run)(tf_tensor[k]) for k in range(len(tf_tensor)))

    probabilities = torch.squeeze(torch.stack(probabilities))

    return probabilities.cpu().numpy()


def preprocess_data(raw, viewing_raw=None):
    """Preprocesses an EEG recording.

    Args:
        raw (mne.io.Raw): EEG recording as mne.Raw object.
        viewing_raw (mne.io.Raw, optional): Preprocessed raw object for plotting. Defaults to None.

    Returns:
        Tuple(torch.Tensor, mne.Epochs, list, float): Transformed to TF images EEG segments, EEG segments, a list of selected channels used by the model, sampling frequency of model output.
    """
    # Segmentation parameters
    windowSize = 1.0  # in seconds
    windowOverlap = 0.5   # ratio overlap between consecutive windows

    preprocessedRaw = raw  # .copy()

    # Remove annotations (but not the annotation intervals!)
    data = preprocessedRaw.get_data()
    new_preprocessedRaw = mne.io.RawArray(data, preprocessedRaw.info)

    # Preprocess
    # Band-pass filter
    new_preprocessedRaw = _preprocessing.filter_fir(new_preprocessedRaw, hp=0.5, lp=45)
    # Interpolate bad channels
    new_preprocessedRaw = _preprocessing.interpolate_bads(new_preprocessedRaw)
    # Re-reference to average electrode
    new_preprocessedRaw = _preprocessing.reref(new_preprocessedRaw, reference='average')
    
    # Re-sample if necessary
    if viewing_raw:
        resample_frequency = viewing_raw.info['sfreq']
        new_preprocessedRaw = _preprocessing.resample(new_preprocessedRaw, resample_frequency)
    
    sample_rate = new_preprocessedRaw.info['sfreq']

    # Pick 19 channels
    selected_channels = _get_10_20_channels(new_preprocessedRaw.ch_names)

    new_preprocessedRaw.pick_channels(selected_channels, ordered=True)
    print("Selected_channels", selected_channels)
    # Segment into epochs of 1 second with chosen ratio overlap
    print("Segmenting EEG ...")
    print("Window size = {a}, window overlap = {b}".format(a=windowSize, b=windowOverlap))
    segmentsRaw = _segmentation.segmentRaw(new_preprocessedRaw, windowSize=windowSize, windowOverlap=windowOverlap)
    print("Number of segments = {}".format(len(segmentsRaw.events)))
    # Get time-frequency values of normalized power for each segment
    print("Converting voltage EEG segments to TF images ...")
    segmentsTF = _tf.segmentTF(segmentsRaw)
    # Convert to image objects
    imgs = [Image.fromarray(np.uint8(segment), mode='L') for segment in segmentsTF]
    # Transform to tensors
    segmentTF_transformed = _transform(imgs)

    return segmentTF_transformed, segmentsRaw, selected_channels, sample_rate


def feed_data_to_model(TF_data, segmentsRaw, model):
    """Feeds data to the CNN model.

    Args:
        TF_data (torch.Tensor): Transformed to TF images EEG segments.
        segmentsRaw (mne.Epochs): EEG segments as mne.Epochs.
        model (Network.ConvNet): Trained CNN with loaded weights.

    Returns:
        predictions (numpy.array): CNN predictions as artifact probability for each data point of EEG recording.
    """
    # Predict
    probabilities = _run_model(model, TF_data)
    segmentsRaw._metadata.reset_index(inplace=True)
    times = [np.arange(segmentsRaw._metadata['start'].tolist()[i],
                       segmentsRaw._metadata['end'].tolist()[i] + 1 / segmentsRaw._metadata['sfreq'].tolist()[i],
                       1 / segmentsRaw._metadata['sfreq'].tolist()[i]) for i in segmentsRaw._metadata.index.tolist()]

    # Smooth predictions
    print("Interpolating and smoothing predictions ...")
    predictions = _smoothing.smooth(probabilities, times)
    print(predictions['probability'].to_numpy())
    print("Length of predictions = {}".format(len(predictions)))

    return predictions['probability'].to_numpy()


def run_model(raw, viewing_raw=None):
    """Runs the CNN model on the time-frequency-transformed EEG data.

    Args:
        raw (mne.io.Raw): Unpreprocessed raw object to get data from.
        viewing_raw (mne.io.Raw, optional): Preprocessed raw object for plotting. Defaults to None.

    Returns:
        Tuple(array, list, float): Array of model output, list of strings of channel names used by the model, sampling frequency of model output.
    """
    TF_data, segmentsRaw, selected_channel_names, sample_rate = preprocess_data(raw, viewing_raw)
    model = load_model()
    model_output = feed_data_to_model(TF_data, segmentsRaw, model)

    return model_output, selected_channel_names, sample_rate, globals.model_annotation_label
