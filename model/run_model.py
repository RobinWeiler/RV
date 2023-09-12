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

import constants as c
import globals

# Standard 10-20 alphabetic channel names
STANDARD_10_20 = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F8', 'T4', 'T6', 'F4', 'C4',
                 'P4', 'O2', 'Fz', 'Cz', 'Pz']

# Biosemi64 standard 10-20 channel names
BIOSEMI64_10_20 = ['Fp1', 'F7', 'T7', 'P7', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F8', 'T8', 'P8', 'F4', 'C4',
                 'P4', 'O2', 'Fz', 'Cz', 'Pz']

# TUAR 10-20 channel names
TUAR_CHANNELS = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
                 'EEG O1-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG F4-REF',
                 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

EGI128_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124',
                'E104', 'E92', 'E83', 'E11', 'Cz', 'E62']

EGI129_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124',
                'E104', 'E92', 'E83', 'E11', 'E129', 'E62']

EGI128_2_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124',
                'E104', 'E92', 'E83', 'E11', 'E55', 'E62']

ADJACENT_10_20 = ['E18','E27','E46','E59','E27','E30','E51','E71','E15','E123','E100','E91','E4','E103','E86','E84','E16','E55','E68']

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
    selected_channels = []
    if all(channel in new_preprocessedRaw.ch_names for channel in STANDARD_10_20):
        selected_channels = STANDARD_10_20
    elif all(channel in new_preprocessedRaw.ch_names for channel in BIOSEMI64_10_20):
        selected_channels = BIOSEMI64_10_20
    elif all(channel in new_preprocessedRaw.ch_names for channel in TUAR_CHANNELS):
        selected_channels = TUAR_CHANNELS
    elif all(channel in new_preprocessedRaw.ch_names for channel in EGI128_10_20):
        selected_channels = EGI128_10_20
    elif all(channel in new_preprocessedRaw.ch_names for channel in EGI128_2_10_20):
        selected_channels = EGI128_2_10_20
    elif all(channel in new_preprocessedRaw.ch_names for channel in EGI129_10_20):
        selected_channels = EGI129_10_20
    else:
        selected_channels = ADJACENT_10_20

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
