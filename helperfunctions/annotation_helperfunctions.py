import mne
import numpy as np

import constants as c


def annotations_to_raw(raw, marked_annotations):
    """Adds marked_annotations to raw object. Change "ANNOTATION_DESCRIPTION" in "constants.py" or here to give annotations different description (currently: "bad_artifact").

    Args:
        raw (mne.io.Raw): Raw object to add annotations to.
        marked_annotations (list): List of tuples(x0, x1) of annotations.

    Returns:
        mne.io.Raw: Raw object with added annotations.
    """
    onset = []
    duration = []
    descriptions = []
    for annotation in marked_annotations:
        onset.append(annotation[0])
        duration.append(annotation[1] - annotation[0])
        descriptions.append(c.ANNOTATION_DESCRIPTION)

    mne_annotations = mne.Annotations(onset, duration, description=descriptions)

    raw.set_annotations(mne_annotations)

    return raw

def get_annotations(raw):
    """Retrieves annotations of given raw object.

    Args:
        raw (mne.io.Raw): Raw object to retrieve annotations from.

    Returns:
        list: List of tuples(x0, x1) of annotations.
    """
    annotation_starts = raw.annotations.onset
    annotation_ends = annotation_starts + raw.annotations.duration
    
    # print(annotation_starts)
    # print(annotation_ends)

    marked_annotations = []
    for annotation_index in range(len(annotation_starts)):
        annotation_ends[annotation_index] = round(annotation_ends[annotation_index], 3)

        marked_annotations.append((annotation_starts[annotation_index], annotation_ends[annotation_index]))

    merged_annotations = merge_intervals(marked_annotations)

    return merged_annotations

def get_annotations_dict(raw):
    """Retrieves annotations of given raw object and stores them in dictionary.

    Args:
        raw (mne.io.Raw): Raw object to retrieve annotations from.

    Returns:
        dict: Dictionary with keys ['onset', 'duration', 'description'] of annotations.
    """
    marked_annotations = {
        'onset': np.round(raw.annotations.onset, 3),
        'duration': np.round(raw.annotations.duration, 3),
        'description': raw.annotations.description
    }

    return marked_annotations

def merge_intervals(marked_annotations):
    """Returns merged version of given annotations.

    Args:
        marked_annotations (list): List of tuples(x0, x1) of annotations.

    Returns:
        list: marked_annotations merged so tuples with overlap are combined.
    """
    marked_annotations = marked_annotations.copy()

    merge_happened = True
    # Currently endless loop
    while merge_happened:
        merge_happened = False

        for i, element in enumerate(marked_annotations):
            # print(marked_annotations)

            for j, compare_element in enumerate(marked_annotations):
                merged_interval = False

                if element == compare_element:
                    continue

                if compare_element[0] <= element[0] and compare_element[1] >= element[0] and compare_element[1] <= element[1]:
                    merged_interval = (compare_element[0], element[1])
                elif compare_element[0] <= element[1] and compare_element[0] >= element[0] and compare_element[1] >= element[1]:
                    merged_interval = (element[0], compare_element[1])
                elif (compare_element[0] <= element[0] and compare_element[1] >= element[1]): #or (element[0] < compare_element[0] and element[1] > compare_element[1]):
                    merged_interval = (compare_element[0], compare_element[1])

                if merged_interval:
                    marked_annotations[i] = merged_interval
                    marked_annotations[j] = merged_interval

                    merge_happened = True

        # Remove duplicates
        marked_annotations = list(dict.fromkeys(marked_annotations))

        print(marked_annotations)

    return marked_annotations

def confidence_intervals(model_output, low_threshold, high_threshold, timestep):
    """Calculates intervals in model-output between given threshold values.

    Args:
        model_output (array): Model-output after sigmoid function.
        low_threshold (float): Low end of confidence interval.
        high_threshold (float): High end of confidence interval.
        timestep (float): Length of one time-step (in seconds).

    Returns:
        list: List of tuples(beginning, end) (in seconds) of intervals between given thresholds.
    """
    datapoints_greater_equal_low_threshold = np.greater(model_output, low_threshold)  # Returns tensor with booleans that are True for each datapoint > low_threshold
    if high_threshold == 1:
        datapoints_lower_high_threshold = np.less_equal(model_output, high_threshold)
    else:
        datapoints_lower_high_threshold = np.less(model_output, high_threshold)  # Returns tensor with booleans that are True for each datapoint < high_threshold

    datapoints_in_interval = []
    for datapoint in range(0, model_output.shape[0]):
        if np.equal(datapoints_greater_equal_low_threshold[datapoint], True) and np.equal(datapoints_lower_high_threshold[datapoint], True):
            datapoints_in_interval.append(datapoint)

    # print("Amount of datapoints between {} and {} confidence: {}".format(low_threshold, high_threshold, len(intervals)))  # For debugging

    if not datapoints_in_interval:
        return datapoints_in_interval  # If there are no annotations, return empty list
    else:
        confidence_intervals = []
        confidence_intervals = annotation_interval_calc(timestep, datapoints_in_interval)  # .copy()

        return confidence_intervals

def annotation_interval_calc(timestep, interval_datapoints):
    """Calculates successive intervals out of list of datapoint-indexes.

    Args:
        timestep (float): Length of one time-step (in seconds).
        interval_datapoints (list): All datapoint-indexes to be included in intervals.

    Returns:
        list: List of tuples(beginning, end) (in seconds) of each interval.
    """
    intervals = []

    annotation_start = interval_datapoints[0]
    annotation_end = annotation_start
    annotation_length = 1

    for index in range(0, len(interval_datapoints)):

        if index + 1 == len(interval_datapoints):
            break

        if interval_datapoints[index + 1] == interval_datapoints[index] + 1:  # If the next annotation in the list is equal to the next datapoint
            annotation_length = annotation_length + 1
        else:
            annotation_end = annotation_start + annotation_length
            intervals.append((annotation_start * timestep, annotation_end * timestep))
            annotation_start = interval_datapoints[index + 1]
    #         annotation_end = annotation_start + 1
            annotation_length = 1

    annotation_end = interval_datapoints[-1]
    intervals.append((annotation_start * timestep, annotation_end * timestep))

    return intervals
