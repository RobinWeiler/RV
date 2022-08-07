import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.transforms import transforms

import model.suppfunc.Network as Network
import model.suppfunc._config as _config
import model.suppfunc._smoothing as _smoothing
import model.suppfunc._segmentation as _segmentation
import model.suppfunc._tf as _tf
import model.suppfunc._preprocessing as _preprocessing
import constants as c

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def _transform(imagesTF):
    # Convert to torch tensor
    trans1 = transforms.ToTensor()
    trans2 = ReshapeTransform((19, 45, 100))

    tensors = [trans2(trans1(image)) for image in imagesTF]
    return torch.stack(tensors).double()


def load_model():
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
    """_summary_

    Args:
        model (Network.ConvNet): Weights of the CNN model trained on time-frequency-transformed EEG segments.
        tf_tensor (torch.tensor): Time-frequency-transformed EEG segments of normalized power obtained from an EEG recording.

    Returns:
        _type_: _description_
    """
    # Reshape TF tensor (n_segments, n_channels, n_freq, n_times) to (n_segments, 1, n_channels, n_freq, n_times)
    tf_tensor = torch.reshape(tf_tensor, (tf_tensor.shape[0], 1, tf_tensor.shape[1],
                                          tf_tensor.shape[2], tf_tensor.shape[3]))
    # Apply model
    print("Applying tfCNN ...")
    print("Device {}".format(DEVICE))
    model.eval()
    with torch.no_grad():
        probabilities = torch.empty(0, 2).double().to(DEVICE)
        for tf in tf_tensor:
            tf = tf.to(DEVICE)
            # Forward pass
            outputs = model(tf)
            _, predicted = torch.max(outputs, 1)
            prob = F.softmax(outputs, dim=1)
            probabilities = torch.cat((probabilities, prob), 0)
    return probabilities.cpu().numpy()


def preprocess_data(raw, viewing_raw=None):
    """_summary_

    Args:
        raw (mne.io.Raw): Un-preprocessed raw object to get data from.
        viewing_raw (mne.io.Raw, optional): Preprocessed raw object for plotting. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Segmentation parameters
    windowSize = 1.0  # in seconds
    windowOverlap = 0.5   # ratio overlap between consecutive windows

    # Preprocess Raw
    # Band-pass filter
    preprocessedRaw = _preprocessing.filter_fir(raw, hp=0.5, lp=45)
    # Interpolate bad channels
    preprocessedRaw = _preprocessing.interpolate_bads(preprocessedRaw)
    # Re-reference to common average
    preprocessedRaw = _preprocessing.reref(preprocessedRaw, reference='average')
    
    # Re-sample if necessary
    if viewing_raw:
        resample_frequency = viewing_raw.info['sfreq']
        preprocessedRaw = _preprocessing.resample(preprocessedRaw, resample_frequency)
    
    sample_rate = preprocessedRaw.info['sfreq']

    # Pick 19 channels
    selected_channels = []
    if all(channel in raw.ch_names for channel in STANDARD_10_20):
        selected_channels = STANDARD_10_20
    elif all(channel in raw.ch_names for channel in BIOSEMI64_10_20):
        selected_channels = BIOSEMI64_10_20
    elif all(channel in raw.ch_names for channel in TUAR_CHANNELS):
        selected_channels = TUAR_CHANNELS
    elif all(channel in raw.ch_names for channel in EGI128_10_20):
        selected_channels = EGI128_10_20
    else:
        selected_channels = raw.ch_names

    raw.pick_channels(selected_channels, ordered=True)
    print("Selected_channels", selected_channels)
    # Segment into epochs of 1 second with chosen ratio overlap
    print("Segmenting EEG ...")
    print("Window size = {a}, window overlap = {b}".format(a=windowSize, b=windowOverlap))
    segmentsRaw = _segmentation.segmentRaw(preprocessedRaw, windowSize=windowSize, windowOverlap=windowOverlap)
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
    """_summary_

    Args:
        TF_data (_type_): _description_
        segmentsRaw (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
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
    """Run CNN model on the time-frequency-transformed EEG data.

    Args:
        raw (mne.io.Raw): Un-preprocessed raw object to get data from.
        viewing_raw (mne.io.Raw, optional): Preprocessed raw object for plotting. Defaults to None.

    Returns:
        Tuple(array, list, float): Array of model-output, list of strings of channel names used by model, sample frequency of model-output.
    """
    TF_data, segmentsRaw, selected_channel_names, sample_rate = preprocess_data(raw, viewing_raw)
    model = load_model()
    model_output = feed_data_to_model(TF_data, segmentsRaw, model)

    return model_output, selected_channel_names, sample_rate

