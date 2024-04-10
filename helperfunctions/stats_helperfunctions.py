import re

from plotly.graph_objs import Figure, Histogram, Scattergl

def __atoi(text):
    return int(text) if text.isdigit() else text

def _natural_keys(text):
    return [ __atoi(c) for c in re.split(r'(\d+)', text) ]

def _get_amount_annotated_clean_data(annotations, recording_length):
    """Calculates amount of annotated- and clean data.

    Args:
        annotations (list): List of tuples(x0, x1) of annotations.
        recording_length (float): Length of recording (in seconds).

    Returns:
        tuple(float, float): Amount of annotated data (in seconds), amount of un-annotated data (in seconds).
    """
    if not annotations:
        return 0.0, recording_length
    else:
        annotations.sort(key=lambda x: x[0])

        amount_annotated_data = 0.0
        amount_annotated_data += annotations[0][1] - annotations[0][0]

        for annotation in annotations[1:]:
            annotation_length = annotation[1] - annotation[0]
            amount_annotated_data += annotation_length

        amount_clean_data = recording_length - amount_annotated_data

        return amount_annotated_data, amount_clean_data

def _get_clean_intervals(annotations, recording_length, interval_length=2):
    if not annotations:
        return [recording_length], (recording_length // interval_length)
    else:
        annotations.sort(key=lambda x: x[0])

        clean_interval_lengths = []
        amount_clean_intervals = 0
        last_annotation_end = 0

        for annotation_starts, annotation_ends, _ in annotations:
            gap = annotation_starts - last_annotation_end
            gap = round(gap)
            if gap > 0:
                clean_interval_lengths.append(gap)

            amount_clean_intervals += max(0, (gap - interval_length)) // interval_length
            last_annotation_end = annotation_ends

        # Check the remaining time after the last event
        remaining_time = recording_length - last_annotation_end
        remaining_time = round(remaining_time)
        if remaining_time > 0:
            clean_interval_lengths.append(remaining_time)

        amount_clean_intervals += max(0, (remaining_time - interval_length)) // interval_length

        return clean_interval_lengths, amount_clean_intervals

def _get_annotated_overlap(annotations1, annotations2):
    if not annotations1 or not annotations2:
        return None
    else:
        annotations1.sort(key=lambda x: x[0])
        annotations2.sort(key=lambda x: x[0])

        amount_annotated_overlap = 0

        annotated_segments1 = []
        annotated_segments2 = []

        for annotation in annotations1:
            start_time, end_time, _ = annotation
            start_time, end_time = round(start_time, 1), round(end_time, 1)

            while start_time + 0.1 <= end_time:
                annotated_segments1.append(start_time)
                start_time += 0.1
                start_time = round(start_time, 1)

        for annotation in annotations2:
            start_time, end_time, _ = annotation
            start_time, end_time = round(start_time, 1), round(end_time, 1)

            while start_time + 0.1 <= end_time:
                annotated_segments2.append(start_time)
                start_time += 0.1
                start_time = round(start_time, 1)

        # print(annotated_segments1)
        # print(len(annotated_segments1))
        # print(annotated_segments2)
        # print(len(annotated_segments2))

        common_segments = set(annotated_segments1) & set(annotated_segments2)
        # print(common_segments)
        # print(len(common_segments))

        unique_segments = set(annotated_segments1) ^ set(annotated_segments2)
        # print(unique_segments)
        # print(len(unique_segments))

        amount_annotated_overlap = (len(common_segments) / (len(common_segments) + len(unique_segments))) * 100
        amount_annotated_overlap = round(amount_annotated_overlap)

        return amount_annotated_overlap

# def _get_annotated_overlap(annotations):
#     if not annotations:
#         return 0.0
#     else:
#         annotations.sort(key=lambda x: x[0])

#         amount_annotated_overlap = 0.0
#         current_annotation = annotations[0]

#         for annotation in annotations[1:]:
#             start_time, end_time, _ = annotation
#             current_start, current_end, _ = current_annotation
            
#             # Check for overlap between the current annotation and the next annotation
#             overlap_start = max(current_start, start_time)
#             overlap_end = min(current_end, end_time)
            
#             # If there is overlap, add it to the total overlap time
#             if overlap_start < overlap_end:
#                 amount_annotated_overlap += overlap_end - overlap_start
            
#             # Update the current annotation if needed
#             if end_time > current_end:
#                 current_annotation = annotation

#         return amount_annotated_overlap

# def _calc_stats(annotations, recording_length, interval_length=2):
#     """Calculates statistics surrounding amounts of annotated- and clean data.

#     Args:
#         annotations (list): List of tuples(x0, x1) of annotations.
#         recording_length (float): Length of recording (in seconds).

#     Returns:
#         tuple(float, float, int, list): Amount of annotated data (in seconds), amount of un-annotated data (in seconds), num un-annotated intervals longer than 2 seconds, list of all lengths of un-annotated intervals (in seconds).
#     """
#     amount_annotated_data, amount_clean_data = _get_amount_annotated_clean_data(annotations, recording_length)
#     clean_interval_lengths, amount_clean_intervals = _get_clean_intervals(annotations, recording_length, interval_length=interval_length)
#     amount_annotated_overlap = _get_annotated_overlap(annotations)

#     return amount_annotated_data, amount_clean_data, amount_clean_intervals, clean_interval_lengths, amount_annotated_overlap

def get_clean_intervals_graph(clean_interval_lengths, recording_length):
    """Generates histogram of given un-annotated interval lengths.

    Args:
        clean_interval_lengths (list): List of all lengths of un-annotated intervals (in seconds)
        recording_length (float): Length of recording.

    Returns:
        plotly.graph_objs.Figure: Histogram of un-annotated interval lengths.
    """
    graph = Figure()

    graph.add_trace(
        Histogram(
            x=clean_interval_lengths,
            xbins=dict(
                start=0,
                end=recording_length,
                size=2
            ),
            # nbinsx=int(recording_length // 2),
            # autobinx=False,
            hovertemplate='Length (in seconds)=%{x}, Amount of intervals=%{y}' + '<extra></extra>',
            marker_color='#4796c5'
        )
    )

    graph.update_layout(
        # plot_bgcolor='white',
        title=dict(
            text='Clean interval lengths',
            # y=0.9,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title_text='Length (in seconds)',
        ),
        yaxis=dict(
            title_text='Amount of intervals'
        ),
        margin=dict(
            pad=0,
            autoexpand=False
        ),
    )

    return graph

def get_power_spectrum_plot(f, all_Pxx_den, channel_names, mean_Pxx_den=[]):
    """Generates power spectrum plot.

    Args:
        f (array): Sample frequencies.
        Pxx_den (array): Power densities.

    Returns:
        plotly.graph_objs.Figure: Plot of power spectrum.
    """
    fig = Figure()

    if len(mean_Pxx_den) == len(f):
        fig.add_trace(
            Scattergl(
                x=f,
                y=mean_Pxx_den,
                name='Mean',
                hovertemplate='<b>%{fullData.name}</b> | Frequency = %{x:.1f} Hz, Power density = %{y:.1f} V**2/Hz' + '<extra></extra>',
                marker=dict(color='black')
            )
        )

    for i, Pxx_den in enumerate(all_Pxx_den):
        fig.add_trace(
            Scattergl(
                x=f,
                y=Pxx_den,
                name=channel_names[i],
                hovertemplate='<b>%{fullData.name}</b> | Frequency = %{x:.1f} Hz, Power density = %{y:.1f} V**2/Hz' + '<extra></extra>',
                opacity=0.6
                # marker=dict(color='#4796c5')
            )
        )

    fig.update_layout(
        xaxis=dict(
            title_text='Frequencies (in Hz)',
        ),
        yaxis=dict(
            title_text='Power spectral density (in V**2/Hz)'
        ),
    )

    return fig
