import numpy as np
import pandas as pd
import math
from scipy import interpolate


def _interpolate(segment_probabilities, segment_times, sfreq, influence=0.5):
    """Returns CNN predictions interpolated for each sample of the EEG signal

    Args:
        segment_probabilities (numpy.array): Predicted class probabilities.
        segment_times (list): Time intervals of EEG segments.
        sfreq (float): Sampling frequency in Hz.
        influence (float): Parameter of influence of past and future predictions (how many segments from past and future to take into account when interpolating CNN predictions in the current segment).

    Returns:
        Tuple (list, list): CNN predictions per EEG sample, sample points of predictions.
    """
    probabilities_per_sample = []
    times = []
    windowSize = int(segment_times[0][-1]-segment_times[0][0]+1/sfreq)
    windowOverlap = segment_times[0][-1] - segment_times[1][0] + 1/sfreq
    windowOffset = windowSize * (1 - windowOverlap)

    # Number of segments in each non-overlapping window of size windowSize
    num_segments_total = math.floor(windowSize / windowOffset)
    # Number of segments from past and future for interpolation
    num_segments_future = math.floor((windowSize - influence) / windowOffset) + 1
    num_segments_past = math.floor((windowSize - influence) / windowOffset)

    t = 0.0  # in s
    # Interpolate first window (no past)
    probabilities = [p[0] for p in segment_probabilities[0:num_segments_future]]
    if num_segments_future > 1:
        x = np.linspace(t, t + windowSize, num=num_segments_future)
        x_int = np.linspace(x[0], x[-1], windowSize * sfreq)  # Linear space with specified number of samples
        f = interpolate.interp1d(x, probabilities, kind='linear')
        y_int = f(x_int)
    else:
        x_int = np.linspace(t, t + windowSize, windowSize * sfreq)  # Linear space with specified number of samples
        y_int = probabilities*len(x_int)

    times.extend(x_int)
    probabilities_per_sample.extend(y_int)
    t = t + windowSize
    # Interpolate for each subsequent window until the one before last
    for i in range(num_segments_total, len(segment_probabilities) - 1, num_segments_total):
        probabilities = [p[0] for p in segment_probabilities[i - num_segments_past:i + num_segments_future]]
        if len(probabilities) > 1:
            x = np.linspace(t, t + windowSize, num=num_segments_past + num_segments_future)
            f = interpolate.interp1d(x, probabilities, kind='linear')
            x_int = np.linspace(x[0], x[-1], windowSize * sfreq)  # Linear space with specified number of samples
            y_int = f(x_int)
        else:
            x_int = np.linspace(t, t + windowSize, windowSize * sfreq)  # Linear space with specified number of samples
            y_int = probabilities*len(x_int)
        times.extend(x_int)
        probabilities_per_sample.extend(y_int)
        t = t + windowSize

    # Interpolate the last window
    probabilities = [p[0] for p in segment_probabilities[len(segment_probabilities) -
                                                         num_segments_past - 1:len(segment_probabilities)]]
    if num_segments_past > 0:
        x = np.linspace(t, t + windowSize, num=len(probabilities))
        f = interpolate.interp1d(x, probabilities, kind='linear')
        x_int = np.linspace(x[0], x[-1], windowSize * sfreq)  # Linear space with specified number of samples
        y_int = f(x_int)
    else:
        x_int = np.linspace(t, t + windowSize, windowSize * sfreq)  # Linear space with specified number of samples
        y_int = probabilities * len(x_int)

    times.extend(x_int)
    probabilities_per_sample.extend(y_int)

    return probabilities_per_sample, times


def _smooth(sample_probabilities, sfreq, smoothing_window=0.25):
    """Returns smoothed CNN predictions.

    Args:
        sample_probabilities (list): Probabilities per EEG sample.
        sfreq (float): Sampling frequency in Hz.
        smoothing_window (float): Smoothing window in seconds.

    Returns:
        sample_probabilities (list): Smoothed CNN predictions.
    """

    smoothing_samples = int(sfreq * smoothing_window)

    artifact_idx = []
    for i, p in enumerate(sample_probabilities):
        if p >= 0.5:
            artifact_idx.append(i)

    # Find indices of artifacts that are closer than smoothing_samples to nearby artifacts
    idxs = [i for i, sample in enumerate(np.diff(artifact_idx)) if (sample > 1 and sample <= smoothing_samples)]
    for i, _ in enumerate(idxs):
        smoothing_region = artifact_idx[idxs[i]:idxs[i] + 2]
        average_probability = (sample_probabilities[smoothing_region[0]] + sample_probabilities[smoothing_region[1]]) / 2
        sample_probabilities[smoothing_region[0]:smoothing_region[1]+1] = \
            [average_probability] * len(sample_probabilities[smoothing_region[0]:smoothing_region[1]+1])

    return sample_probabilities


def smooth(segment_probabilities, segment_times):
    """Returns CNN predictions as artifact probability per EEG sample.

    Args:
        segment_probabilities (numpy.array): Predicted class probabilities.
        segment_times (list): Time intervals of EEG segments.

    Returns:
        df (pandas.DataFrame): Dataframe of CNN predictions per EEG sample.
    """
    # Sampling frequency in Hz
    sfreq = int(1/(segment_times[0][1]-segment_times[0][0]))
    probabilities_per_sample, times = _interpolate(segment_probabilities, segment_times, sfreq)
    smoothed_probabilities = _smooth(probabilities_per_sample, sfreq)

    df = pd.DataFrame(list(smoothed_probabilities), index=range(len(smoothed_probabilities)),
                      columns=['probability'])
    return df