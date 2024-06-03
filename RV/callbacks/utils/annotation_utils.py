import random

from dash import dcc, html

import mne

import RV.constants as c


# Add 'hide' to color options
COLOR_OPTIONS = c.ANNOTATION_COLOR_OPTIONS.copy()
COLOR_OPTIONS.insert(0, 'hide')  # allows to hide annotations of a given label


def get_annotation_label_radioitem(annotation_label: str, color=None):
    """Generates dictionary with annotation-label option for dcc.RadioItems.

    Args:
        annotation_label (str): Annotation label for which to generate a radioitem.
        color (str, optional): Color (from COLOR_OPTIONS) to use for given annotation label. Defaults to None (random choice).

    Returns:
        dict: Dictionary with keys 'label' and 'value'.
    """
    if not color:
        # random color if no color was given
        color = random.choice(c.ANNOTATION_COLOR_OPTIONS)

    annotation_label_dict = {
        'label':[
            html.Span(annotation_label),
            dcc.Dropdown(
                options=[{'label': f'{color}', 'value': f'{color}'} for color in COLOR_OPTIONS],
                value=color,
                clearable=False,
                className='small-dropdown',
                id={'type': 'color-dropdown', 'label': annotation_label}
            )
        ],
        'value': annotation_label
    }

    return annotation_label_dict, color

def merge_annotations(marked_annotations: list):
    """Merges list of marked annotations if there is overlap. 

    Args:
        marked_annotations (list): A list of tuples (annotation_onset, annotation_duration, annotation_label).

    Returns:
        list: List of merged tuples (annotation_onset, annotation_duration, annotation_label).
        bool: Whether or not a merge took place.
    """
    if len(marked_annotations) < 2:
        return marked_annotations, False

    annotations_copy = marked_annotations.copy()
    merge_happened = True

    # Sort annotations by onset
    annotations_copy = sorted(annotations_copy, key=lambda annotation: annotation[0])

    # Iterates through all annotations until nothing was merged
    while merge_happened and len(annotations_copy) > 1:
        merged_annotations = annotations_copy.copy()

        for index in range(len(annotations_copy)):
            if index < len(annotations_copy) - 1:
                annotation_end = annotations_copy[index][0] + annotations_copy[index][1]

                # if the next annotation starts within the current one and they have the same label
                if (annotations_copy[index + 1][0] <= annotation_end) and (annotations_copy[index][2] == annotations_copy[index + 1][2]):
                    annotation_end_2 = annotations_copy[index + 1][0] + annotations_copy[index + 1][1]

                    # if the next annotation ends later than the current one, extend the duration of the current one
                    if annotation_end_2 > annotation_end:
                        merged_annotations[index] = (merged_annotations[index][0], annotation_end_2 - merged_annotations[index][0], merged_annotations[index][2])

                    # remove the next annotation and restart the loop
                    merged_annotations.pop(index + 1)
                    annotations_copy = merged_annotations.copy()
                    merge_happened = True
                    break

                else:
                    merge_happened = False

    if len(marked_annotations) != len(merged_annotations):
        merge_happened = True

    return merged_annotations, merge_happened

def annotations_to_raw(marked_annotations: list, raw: mne.io.Raw):
    """Update annotations in given mne.io.Raw object with given marked_annotations.

    Args:
        marked_annotations (list): A list of tuples (annotation_onset, annotation_duration, annotation_label).
        raw (mne.io.Raw): Raw object holding EEG data.

    Returns:
        mne.io.Raw: Raw object with given annotations.
    """
    onsets = []
    durations = []
    descriptions = []
    for annotation in marked_annotations:
        onsets.append(annotation[0])
        durations.append(annotation[1])
        descriptions.append(annotation[2])

    mne_annotations = mne.Annotations(onsets, durations, description=descriptions)
    raw.set_annotations(mne_annotations)

    return raw
