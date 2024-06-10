from dash import dcc, html

import plotly.graph_objs as go

import mne

import RV.constants as c
from RV.callbacks.utils.annotation_utils import merge_annotations
from RV.callbacks.utils.bad_channels_utils import bad_channel_disagrees


def get_amount_annotated_data(merged_annotations: list):
    """Calculates amount of annotated data in seconds.

    Args:
        merged_annotations (list of (annotation_onset, annotation_duration)): Merged annotations (regardless of label) in recording.

    Returns:
        float: Amount of annotated data (in seconds).
    """
    if len(merged_annotations) == 0:
        return 0.0
    else:
        amount_annotated_data = 0.0
        for annotation in merged_annotations:
            amount_annotated_data += annotation[1]

        return amount_annotated_data

def get_annotated_100_milliseconds(annotations):
    """Retrieves timepoints (in 100 milliseconds) contained within given annotations.

    Args:
        annotations (list of (annotation_onset, annotation_duration)): Annotation intervals.

    Returns:
        list: List of 100-millisecond timepoints.
    """
    annotated_timepoints = []

    for annotation in annotations:
        onset, duration, _ = annotation
        onset, end_time = round(onset, 1), round((onset + duration), 1)

        while onset + 0.1 <= end_time:
            annotated_timepoints.append(onset)
            onset += 0.1
            onset = round(onset, 1)

    return annotated_timepoints

def get_percentage_annotation_overlap(annotations, other_annotations):
    """Calculates percentage of overlap (in 100-millisecond windows) between intervals in annotations and other_annotations.

    Args:
        annotations (list of (annotation_onset, annotation_duration)): Annotation intervals of certain label.
        other_annotations (list of (annotation_onset, annotation_duration)): Annotation intervals of certain other label.

    Returns:
        float: Percentage of overlap between intervals.
    """
    if len(annotations) == 0 or len(other_annotations) == 0:
        return None

    # Sort by onset
    annotations.sort(key=lambda x: x[0])
    other_annotations.sort(key=lambda x: x[0])

    annotated_timepoints = get_annotated_100_milliseconds(annotations)
    other_annotated_timepoints = get_annotated_100_milliseconds(other_annotations)

    if len(annotated_timepoints) == 0 or len(other_annotated_timepoints) == 0:
        return None

    common_timepoints = set(annotated_timepoints) & set(other_annotated_timepoints)
    unique_timepoints = set(annotated_timepoints) ^ set(other_annotated_timepoints)

    percentage_annotation_overlap = (len(common_timepoints) / (len(common_timepoints) + len(unique_timepoints))) * 100

    return percentage_annotation_overlap

def get_non_annotated_intervals(merged_annotations: list, recording_length: float):
    """Generates list of non-annotated data intervals.

    Args:
        merged_annotations (list of (annotation-onset, annotation-duration)): Merged annotations (regardless of label) in recording.
        recording_length (float): Length of EEG recording (in seconds).

    Returns:
        list of (non-annotated-onset, non-annotated-duration): Non-annotated data intervals.
    """
    if len(merged_annotations) == 0:
        return [(0.0, recording_length)]
    else:
        # Sort by onset
        merged_annotations = sorted(merged_annotations, key=lambda annotation: annotation[0])

        non_annotated_intervals = []
        previous_annotation_end = 0.0

        for annotation_start, annotation_duration, _ in merged_annotations:
            non_annotated_duration = annotation_start - previous_annotation_end
            if non_annotated_duration > 0:
                non_annotated_intervals.append((previous_annotation_end, non_annotated_duration))

            previous_annotation_end = annotation_start + annotation_duration

        # Add remaining time after the last annotation
        remaining_time = recording_length - previous_annotation_end
        if remaining_time > 0:
            non_annotated_intervals.append((previous_annotation_end, remaining_time))

        return non_annotated_intervals

def get_num_segments_of_length(intervals: list, interval_length: float):
    """Calculates amount of segments of length interval_length in given intervals.

    Args:
        intervals (list of (onset, duration)): List of time intervals.
        interval_length (float): Length of segments to count.

    Returns:
        int: Amount of segments of length interval_length.
    """
    num_intervals = 0

    if len(intervals) > 0:
        # Sort by onset
        intervals = sorted(intervals, key=lambda interval: interval[0])

        for interval_start, interval_duration in intervals:
            num_intervals += int(interval_duration // interval_length)

    return num_intervals

def get_interval_durations_histogram(intervals: list, recording_length: float):
    """Generates histogram of durations of given intervals.

    Args:
        intervals (list): List of tuples of onset and duration of intervals (in seconds).
        recording_length (float): Length of EEG recording (in seconds).

    Returns:
        plotly.graph_objs.Figure: Histogram of interval durations.
    """
    durations = [interval[1] for interval in intervals]

    histogram = go.Figure()

    if len(durations) > 0:
        histogram.add_trace(
            go.Histogram(
                x=durations,
                xbins=dict(
                    start=0,
                    end=recording_length,
                    size=2
                ),
                # nbinsx=int(recording_length // 2),
                # autobinx=False,
                hovertemplate='Length (in seconds)=%{x}, Amount of intervals=%{y}' + '<extra></extra>',
                marker_color='#2bb1d6'
            )
        )

    histogram.update_layout(
        margin=dict(
            autoexpand=False,
            t=0,
            pad=0,
        ),
        plot_bgcolor=c.PLOT_COLOR,
    )

    histogram.update_xaxes(
        title=dict(text='Length (in seconds)')
    )

    histogram.update_yaxes(
        title=dict(text='Amount of intervals')
    )

    return histogram

def get_annotation_stats(raw: mne.io.Raw, recording_length: float, annotation_labels: list):
    """Generates dash.html.Div with statistics surrounding annotations of given mne.io.Raw object.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        recording_length (float): Length of EEG recording (in seconds).
        annotation_label_options (list): List of strings of annotation labels.

    Returns:
        dash.html.Div: Div containing annotation statistics.
    """
    marked_annotations = []
    merged_annotations = []

    for annotation_index in range(len(raw.annotations)):
        marked_annotations.append((raw.annotations.onset[annotation_index], raw.annotations.duration[annotation_index], raw.annotations.description[annotation_index]))

        # Merge all annotations regardless of label for total amount of annotated data
        merged_annotations.append((raw.annotations.onset[annotation_index], raw.annotations.duration[annotation_index], 'temp'))
    merged_annotations, _ = merge_annotations(merged_annotations)

    total_amount_annotated_data = get_amount_annotated_data(merged_annotations)
    
    annotated_data_stats = html.Div([
        html.H1('Annotated data'),

        html.Div([
            html.H2('Total amount of annotated data:'),
            html.Span(f'{round(total_amount_annotated_data, 1)} seconds')
        ]),
    ])

    annotations_per_label = []
    for annotation_label in annotation_labels:
        corresponding_annotations = [(annotation[0], annotation[1], annotation[2]) for annotation in marked_annotations if annotation[2] == annotation_label]
        annotations_per_label.append(corresponding_annotations)

        amount_annotated_data = get_amount_annotated_data(corresponding_annotations)
        percentage_annotated_data = (amount_annotated_data / recording_length) * 100

        annotated_data_stats.children.append(
            html.Div([
                html.H2(f"Amount of annotated data labeled '{annotation_label}':"),
                html.Span(f'{round(amount_annotated_data, 1)} seconds ({round(percentage_annotated_data)}% of recording)')
            ]),
        )

    # Compare each annotation label to each other annotation label
    for annotation_index in range(len(annotations_per_label) - 1):
        if len(annotations_per_label[annotation_index]) == 0:
            continue

        # Avoid comparing the same annotation labels twic
        for other_annotation_index in range(annotation_index + 1, len(annotations_per_label)):
            if len(annotations_per_label[other_annotation_index]) == 0:
                continue

            if annotations_per_label[annotation_index][0][2] == annotations_per_label[other_annotation_index][0][2]:
                # Skip annotations with the same label
                continue

            percentage_annotation_overlap = get_percentage_annotation_overlap(annotations_per_label[annotation_index], annotations_per_label[other_annotation_index])
            annotated_data_stats.children.append(
                html.Div([
                    html.H2(f"Amount of overlap between annotations labeled '{annotations_per_label[annotation_index][0][2]}' and '{annotations_per_label[other_annotation_index][0][2]:}':"),
                    html.Span(str(round(percentage_annotation_overlap) if percentage_annotation_overlap != None else 0) + '%')
                ]),
            )

    # Calculate statistics surrounding non-annotated data
    total_amount_non_annotated_data = recording_length - total_amount_annotated_data
    percentage_non_annotated_data = (total_amount_non_annotated_data / recording_length) * 100

    non_annotated_intervals = get_non_annotated_intervals(merged_annotations, recording_length)
    non_annotated_intervals_2_seconds = get_num_segments_of_length(non_annotated_intervals, interval_length=2)
    non_annotated_durations_histogram = get_interval_durations_histogram(non_annotated_intervals, recording_length)

    non_annotated_data_stats = html.Div([
        html.H1('Non-annotated data'),

        html.Div([
            html.H2('Total amount of non-annotated data left:'),
            html.Span(f'{round(total_amount_non_annotated_data, 1)} seconds ({round(percentage_non_annotated_data)}% of recording)')
        ]),
        html.Div([
            html.H2('Total amount of non-annotated intervals longer than 2 seconds:'),
            html.Span(str(non_annotated_intervals_2_seconds))
        ]),
        html.Div([
            html.H2('Non-annotated interval lengths:'),
            dcc.Graph(
                id='RV-non-annotated-intervals-histogram-graph',
                figure=non_annotated_durations_histogram,
                config={
                    'displayModeBar': False,
                },
            ),
        ]),
    ])

    annotation_stats = html.Div([
        non_annotated_data_stats,
        html.Hr(),

        annotated_data_stats
    ])

    return annotation_stats

def get_bad_channel_info(selected_bad_channels: list, bad_channels_dict: dict):
    """Generates dash.html.Div with info surrounding selected bad channels.

    Args:
        selected_bad_channels (list): List of strings of bad-channel names.
        bad_channels_dict (dict): Dictionary with source of bad channels as key and bad channels as values.

    Returns:
        dash.html.Div: Div containing bad-channel info.
    """
    disagreed_bad_channels = []
    for channel_name in selected_bad_channels:
        if bad_channel_disagrees(channel_name, bad_channels_dict):
            disagreed_bad_channels.append(channel_name)

    bad_channel_stats = html.Div([
        html.H1('Bad channels'),

        html.Div([
            html.H2('Selected bad channels:'),
            html.Span(', '.join(selected_bad_channels))
        ]),

        html.Div([
            html.H2('Disagreed bad channels:'),
            html.Span(', '.join(disagreed_bad_channels))
        ]),

        html.Div(
            children=[
                html.Div([
                    html.H2(f'Bad channels from {source}:'),
                    html.Span(', '.join(bad_channels))
                ])
            for source, bad_channels in bad_channels_dict.items()]
        )
    ])

    return bad_channel_stats
