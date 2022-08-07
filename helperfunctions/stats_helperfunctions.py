from plotly.graph_objs import Figure, Histogram


def calc_stats(annotations, recording_length):
    """Calculates statistics surrounding amounts of annotated- and clean data.

    Args:
        annotations (list): List of tuples(x0, x1) of annotations.
        recording_length (float): Length of recording (in sec).

    Returns:
        tuple(float, float, int, list): Amount of annotated data (in sec), amount of un-annotated data (in sec), num un-annotated intervals longer than 2 seconds, list of all lengths of un-annotated intervals (in sec).
    """
    if annotations:
        annotations.sort()

    amount_annotated_data = 0.0
    for annotation in annotations:
        annotation_length = annotation[1] - annotation[0]
        # print(annotation_length)
        amount_annotated_data += annotation_length
    # print('#Noisy data: {}'.format(amount_noisy_data))

    amount_clean_data = recording_length - amount_annotated_data
    # print('#Clean data: {}'.format(amount_clean_data))

    # Count amount of clean intervals longer/equal to 2 seconds
    if annotations:
        amount_clean_intervals_2sec = 0
        index = 0
        while index < recording_length:
            if index + 2 > recording_length:
                break
            index += 2

            overlapped = False
            for annotation in annotations:
                if index > annotation[0] and index <= annotation[1]:
                    index = annotation[1]
                    overlapped = True
            if not overlapped:
                amount_clean_intervals_2sec += 1
    else:
        amount_clean_intervals_2sec = recording_length // 2

    clean_interval_lengths = []
    if annotations:
        clean_interval_lengths.append(round(annotations[0][0], 2))  # beginning of recording until first annotation
        for index in range(len(annotations)):
            if index + 1 == len(annotations):
                clean_interval_lengths.append(round(recording_length - annotations[index][1], 2))
                break
            clean_interval_lengths.append(round(annotations[index + 1][0] - annotations[index][1], 2))
    else:
        clean_interval_lengths.append(round(recording_length, 2))
    # print(clean_interval_lengths)

    return amount_annotated_data, amount_clean_data, amount_clean_intervals_2sec, clean_interval_lengths

def get_clean_intervals_graph(clean_interval_lengths, recording_length):
    """Generates histogram of given un-annotated interval lengths.

    Args:
        clean_interval_lengths (list): List of all lengths of un-annotated intervals (in sec)
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
            hovertemplate='Length-range (in sec)=%{x}, Amount of intervals=%{y}' + '<extra></extra>',
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
            title_text='Length (in sec)',
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
