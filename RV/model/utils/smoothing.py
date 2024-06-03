import math

import numpy as np
import pandas as pd

from scipy import interpolate


def interpolate_predictions(segment_probabilities: np.array, segment_times: list, sfreq: float, influence=0.5):
    """Returns CNN predictions interpolated for each sample of the EEG signal.

    Args:
        segment_probabilities (numpy.array): Predicted class probabilities.
        segment_times (list): Time intervals of EEG segments.
        sfreq (float): Sampling frequency (in Hz).
        influence (float): Parameter of influence of past and future predictions (how many segments from past and future to take into account when interpolating CNN predictions in the current segment).

    Returns:
        Tuple (list, list): CNN predictions per EEG sample and sample points of predictions.
    """
    probabilities_per_sample = []
    times = []

    window_size = int(segment_times[0][-1] - segment_times[0][0] + (1 / sfreq))
    window_overlap = segment_times[0][-1] - segment_times[1][0] + (1 / sfreq)
    windowOffset = window_size * (1 - window_overlap)

    # Number of segments in each non-overlapping window of size window_size
    num_segments_total = math.floor(window_size / windowOffset)
    # Number of segments from past and future for interpolation
    num_segments_future = math.floor((window_size - influence) / windowOffset) + 1
    num_segments_past = math.floor((window_size - influence) / windowOffset)

    t = 0.0

    # Interpolate first window (no past)
    probabilities = [p[0] for p in segment_probabilities[0:num_segments_future]]
    if num_segments_future > 1:
        x = np.linspace(t, t + window_size, num=num_segments_future)
        x_int = np.linspace(x[0], x[-1], window_size * sfreq)  # Linear space with specified number of samples
        f = interpolate.interp1d(x, probabilities, kind='linear')
        y_int = f(x_int)
    else:
        x_int = np.linspace(t, t + window_size, window_size * sfreq)  # Linear space with specified number of samples
        y_int = probabilities*len(x_int)

    times.extend(x_int)
    probabilities_per_sample.extend(y_int)
    t = t + window_size

    # Interpolate for each subsequent window until the one before last
    for i in range(num_segments_total, len(segment_probabilities) - 1, num_segments_total):
        probabilities = [p[0] for p in segment_probabilities[i - num_segments_past:i + num_segments_future]]

        if len(probabilities) > 1:
            x = np.linspace(t, t + window_size, num=(num_segments_past + num_segments_future))
            f = interpolate.interp1d(x, probabilities, kind='linear')
            x_int = np.linspace(x[0], x[-1], window_size * sfreq)
            y_int = f(x_int)
        else:
            x_int = np.linspace(t, t + window_size, window_size * sfreq)
            y_int = probabilities * len(x_int)

        times.extend(x_int)
        probabilities_per_sample.extend(y_int)
        t = t + window_size

    # Interpolate the last window
    probabilities = [p[0] for p in segment_probabilities[len(segment_probabilities) - num_segments_past - 1:len(segment_probabilities)]]

    if num_segments_past > 0:
        x = np.linspace(t, t + window_size, num=len(probabilities))
        f = interpolate.interp1d(x, probabilities, kind='linear')

        x_int = np.linspace(x[0], x[-1], window_size * sfreq)
        y_int = f(x_int)
    else:
        x_int = np.linspace(t, t + window_size, window_size * sfreq)
        y_int = probabilities * len(x_int)

    times.extend(x_int)
    probabilities_per_sample.extend(y_int)

    return probabilities_per_sample, times

def smooth_predictions(segment_probabilities: np.array, segment_times: list):
    """Returns smoothed CNN predictions as artifact probability per EEG datapoint.

    Args:
        segment_probabilities (np.array): Predicted class probabilities.
        segment_times (list): Time intervals of EEG segments.

    Returns:
        df (np.arraye): CNN predictions as artifact probability.
    """
    # Sampling frequency in Hz
    sfreq = int(1 / (segment_times[0][1] - segment_times[0][0]))

    probabilities_per_sample, times = interpolate_predictions(segment_probabilities, segment_times, sfreq)

    smoothing_window = 0.25
    smoothing_samples = int(sfreq * smoothing_window)

    artifact_idx = []
    for i, p in enumerate(probabilities_per_sample):
        if p >= 0.5:
            artifact_idx.append(i)

    # Find indices of artifacts that are closer than smoothing_samples to nearby artifacts
    idxs = [i for i, sample in enumerate(np.diff(artifact_idx)) if (sample > 1 and sample <= smoothing_samples)]
    for i, _ in enumerate(idxs):
        smoothing_region = artifact_idx[idxs[i]:idxs[i] + 2]
        average_probability = (probabilities_per_sample[smoothing_region[0]] + probabilities_per_sample[smoothing_region[1]]) / 2
        probabilities_per_sample[smoothing_region[0]:smoothing_region[1]+1] = \
            [average_probability] * len(probabilities_per_sample[smoothing_region[0]:smoothing_region[1]+1])

    df = pd.DataFrame(list(probabilities_per_sample), index=range(len(probabilities_per_sample)),
                      columns=['probability'])

    smoothed_predictions = df['probability'].to_numpy()

    return smoothed_predictions
