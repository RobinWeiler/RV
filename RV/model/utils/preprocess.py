from PIL import Image

import numpy as np
import pandas as pd

import torch
from torchvision.transforms import transforms

import mne

from RV.model.utils import tf
from RV.callbacks.utils.channel_selection_utils import get_10_20_channels


def segment_metadata(raw: mne.io.Raw, events: np.array, window_size: float):
    """Returns dataframe containing metadata for segmentation.

    Args:
       raw (mne.io.Raw): Raw object holding EEG data.
        events (np.array): Events to be used for segmentation.
        window_size (float): Size of window in seconds.

    Returns:
        df (pandas.DataFrame): Dataframe containing metadata of EEG segments.
    """
    event_idx = range(len(events))
    ids = events[:, 2]

    labels = np.where(np.array(ids) == 0, 'artifact', (np.where(np.array(ids) == 2, 'ignored', 'nonartifact')))

    start = [onset[0] / raw.info['sfreq'] for onset in events]
    end = [(onset + window_size - (1 / raw.info['sfreq'])) for onset in start]

    sfreq = [raw.info['sfreq']] * len(events)
    filenames = [raw._filenames[0]] * len(events)

    df = pd.DataFrame(list(zip(start, end, sfreq, labels, filenames)), index=event_idx,
                      columns=['start', 'end', 'sfreq', 'true_label', 'filename'])

    return df

def segment_raw(raw: mne.io.Raw, window_size: float, window_overlap=0.5):
    """Returns segments of an EEG recording as mne.Epochs.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        window_size (float): Size of the segmenting window in seconds.
        window_overlap (float): Overlap between consecutive windows. Defaults to 0.5.

    Returns:
        segments (mne.Epochs): EEG segments as mne.Epochs.
    """
    sfreq = raw.info['sfreq']

    # Epoch length in timepoints
    epoch_length_timepoints = sfreq * window_size

    # Offset
    epoch_offset_seconds = window_size - window_size*window_overlap
    epoch_offset_timepoints = int(sfreq * epoch_offset_seconds)

    # Make a set of events/segments separated by a fixed offset
    n_epochs = int(np.ceil((raw.__len__() - epoch_length_timepoints) / epoch_offset_timepoints + 1))

    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 2] = 1
    events[:, 0] = np.array(np.linspace(0, (n_epochs * epoch_offset_seconds) - epoch_offset_seconds, n_epochs) * sfreq, dtype=int)

    # Create metadata for segments
    metadata = segment_metadata(raw, events, window_size=window_size)

    # Create mne.Epochs based on events
    segments = mne.Epochs(raw, events, event_id={'nonartifact': 1}, preload=True, baseline=None, tmin=0, tmax=(window_size - (1 / sfreq)), metadata=metadata, reject_by_annotation=False, verbose=0)

    return segments

def preprocess_data(raw: mne.io.Raw, device: torch.device):
    """Preprocesses an EEG recording for CNN model.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        Tuple(torch.Tensor, mne.Epochs, list): EEG segments transformed to time-frequency plots in torch.Tensors, EEG segments as mne.Epochs, and list of strings of channel names considered by the model.
    """
    # Segmentation parameters
    window_size = 1.0  # seconds
    window_overlap = 0.5  # ratio overlap between consecutive windows

    # Band-pass filter
    if raw.info['highpass'] != 0.5 or raw.info['lowpass'] != 45:
        raw.filter(0.5, 45, fir_design="firwin", verbose=0)

    # Interpolate bad channels
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=False)

    # Re-reference to average electrode
    if not raw.info['custom_ref_applied']:
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=0)

    # Pick 19 10-20 channels
    selected_channels = get_10_20_channels(raw.ch_names)
    raw.pick(selected_channels)
    print(f'Considered channels: {selected_channels}')

    # Segment into epochs of 1 second with 0.5 overlap
    print('Segmenting EEG ...')
    print(f'Window size = {window_size}, window overlap = {window_overlap}')
    segments = segment_raw(raw, window_size=window_size, window_overlap=window_overlap)
    print("Number of segments = {}".format(len(segments.events)))

    # Get time-frequency values of normalized power for each segment
    print("Converting voltage EEG segments to TF images ...")
    segments_TF = tf.segment_TF(segments, device)

    # Convert to image objects
    images = [Image.fromarray(np.uint8(segment), mode='L') for segment in segments_TF]

    # Transform to tensors
    segments_TF_tensors = [torch.reshape((transforms.ToTensor()(image)), (19, 45, 100)) for image in images]
    segments_TF_tensors = torch.stack(segments_TF_tensors).double()

    return segments_TF_tensors, segments, selected_channels
