
import mne
import pandas as pd
import numpy as np


def _segment_metadata(Raw, events, windowSize):

    event_idx = range(len(events))
    ids = events[:, 2]
    labels = np.where(np.array(ids) == 0, 'artifact', (np.where(np.array(ids) == 2, 'ignored', 'nonartifact')))
    start = [onset[0]/Raw.info['sfreq'] for onset in events]
    end = [onset + windowSize - 1./Raw.info['sfreq'] for onset in start]
    sfreq = [Raw.info['sfreq']]*len(events)
    filenames = [Raw._filenames[0]]*len(events)

    df = pd.DataFrame(list(zip(start, end, sfreq, labels, filenames)), index=event_idx,
                      columns=['start', 'end', 'sfreq', 'true_label', 'filename'])

    return df


def segmentRaw(Raw, windowSize, windowOverlap=0.5):

    """
    Parameters
    ----------
    Raw :

    windowSize :

    windowOverlap :

    Returns
    -------
    """
    sfreq = Raw.info["sfreq"]
    # Epoch length in timepoints
    epoch_length_timepoints = sfreq*windowSize
    # Offset in seconds
    epoch_offset_seconds = windowSize-windowSize*windowOverlap
    # Offset in timepoints
    epoch_offset_timepoints = int(sfreq * epoch_offset_seconds)
    # Make a set of events/segments separated by a fixed offset
    n_epochs = int(np.ceil( (Raw.__len__() - epoch_length_timepoints) / epoch_offset_timepoints + 1) )
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 2] = 1
    events[:, 0] = np.array(
        np.linspace(0, (n_epochs * epoch_offset_seconds) - epoch_offset_seconds, n_epochs) * sfreq,
        dtype=int)
    # Create metadata for segments
    metadata = _segment_metadata(Raw, events, windowSize=windowSize)
    # Create segments/mne epochs based on events
    segments = mne.Epochs(Raw, events, event_id={'nonartifact': 1}, preload=True,
                        tmin=0, tmax=windowSize-1/sfreq, metadata=metadata, baseline=None, verbose=0)

    return segments