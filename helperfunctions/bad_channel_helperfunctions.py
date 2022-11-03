import numpy as np
import pandas as pd

import mne
from autoreject import AutoReject, Ransac

from multiprocessing import cpu_count

import constants as c


def _segmentation(raw, windowSize=c.WINDOW_SIZE, windowOverlap=c.WINDOW_OVERLAP):
    """_summary_

    Args:
        raw (mne.io.Raw): Raw object to segment.
        windowSize (float, optional): _description_. Defaults to c.WINDOW_SIZE.
        windowOverlap (float, optional): _description_. Defaults to c.WINDOW_OVERLAP.

    Returns:
        mne.Epochs: _description_
    """
    sfreq = raw.info["sfreq"]
    # Epoch length in timepoints
    epoch_length_timepoints = sfreq * windowSize
    # Offset in seconds
    epoch_offset_seconds = windowSize - windowSize * windowOverlap
    # Offset in timepoints
    epoch_offset_timepoints = int(sfreq * epoch_offset_seconds)
    # Make a set of events/segments separated by a fixed offset
    n_epochs = int(np.ceil((raw.__len__() - epoch_length_timepoints) / epoch_offset_timepoints + 1))
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 2] = 3  # just the label, can be any integer - doesn't matter
    events[:, 0] = np.array(
        np.linspace(0, (n_epochs * epoch_offset_seconds) - epoch_offset_seconds, n_epochs) * sfreq,
        dtype=int)
    # Create segments/mne epochs based on events
    segments = mne.Epochs(raw, events, event_id={'unknown': 3}, preload=True,
                          tmin=0, tmax=windowSize - 1 / sfreq, baseline=None, verbose=0)
    return segments

def _find_bad_channels_autoreject(segments, n_interpolate=c.N_INTERPOLATE, consensus=c.CONSENSUS):
    """_summary_

    Args:
        segments (mne.Epochs): _description_
        n_interpolate (np.array, optional): _description_. Defaults to c.N_INTERPOLATE.
        consensus (np.linspace, optional): _description_. Defaults to c.CONSENSUS.

    Returns:
        list: List of bad-channel names.
    """
    num_workers = cpu_count()
    print('Number of workers: {}'.format(num_workers))
    
    # Define AR
    ar = AutoReject(n_interpolate=[1], consensus=[0.6], n_jobs=num_workers, random_state=0)
    # Fit to data - do not transform
    ar.fit(segments)
    # Get rejection log
    reject_log = ar.get_reject_log(segments)
    # Chosen parameter values
    consensus_param = ar.consensus_
    n_interpolates_param = ar.n_interpolate_

    # Put into a dataframe (rows are segments, columns are channels, values are labels whether the channels is good (0), bad (1), or interpolated (2) for each segment)
    df_ar = pd.DataFrame(reject_log.labels, columns=reject_log.ch_names)

    # For each channel, calculate the ratio of segments for which the channel is bad or interpolated
    ratio_bad_segments = (df_ar > 0).sum(axis=0).values / len(reject_log.labels)
    # Get channels which are bad based on the threshold
    all_channels = np.array((df_ar > 0).sum(axis=0).index.tolist())
    bad_channels = all_channels[np.where(ratio_bad_segments >= c.THRESHOLD)[0]]

    bad_channels = bad_channels.tolist()

    return bad_channels

def _find_bad_channels_ransac(segments):
    """_summary_

    Args:
        segments (mne.Epochs): _description_

    Returns:
        list: List of bad-channel names.
    """
    num_workers = cpu_count()
    print('Number of workers: {}'.format(num_workers))
    
    # Define AR
    ransac = Ransac(n_jobs=num_workers, random_state=0)
    # Fit to data - do not transform
    ransac.fit(segments)

    bad_channels = ransac.bad_chs_

    return bad_channels

def get_bad_channels(raw, method):
    """_summary_

    Args:
        raw (mne.io.Raw): Raw object to get bad channels from.

    Returns:
        list: List of bad-channel names.
    """
    segments = _segmentation(raw)
    
    if method == 'AutoReject':
        bad_channels = _find_bad_channels_autoreject(segments)
    elif method == 'RANSAC':
        bad_channels = _find_bad_channels_ransac(segments)
    else:
        print('This method is not supported')
        bad_channels = []

    return bad_channels
