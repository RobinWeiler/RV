from plotly.graph_objs import Figure, Histogram, Scattergl

from scipy.signal import welch

def calc_stats(annotations, recording_length):
    """Calculates statistics surrounding amounts of annotated- and clean data.

    Args:
        annotations (list): List of tuples(x0, x1) of annotations.
        recording_length (float): Length of recording (in seconds).

    Returns:
        tuple(float, float, int, list): Amount of annotated data (in seconds), amount of un-annotated data (in seconds), num un-annotated intervals longer than 2 seconds, list of all lengths of un-annotated intervals (in seconds).
    """
    if annotations:
        annotations.sort()

    amount_annotated_data = 0.0
    if annotations:
        amount_annotated_data += (annotations[0][1] - annotations[0][0])
        current_annotation = annotations[0]

    amount_annotated_overlap = 0.0
    for annotation in annotations[1:]:
        annotation_length = annotation[1] - annotation[0]
        amount_annotated_data += annotation_length

        start_time, end_time, _ = annotation
        current_start, current_end, _ = current_annotation
        
        # Check for overlap between the current annotation and the next annotation
        overlap_start = max(current_start, start_time)
        overlap_end = min(current_end, end_time)
        
        # If there is overlap, add it to the total overlap time
        if overlap_start < overlap_end:
            amount_annotated_overlap += overlap_end - overlap_start
        
        # Update the current annotation if needed
        if end_time > current_end:
            current_annotation = annotation

    amount_clean_data = recording_length - amount_annotated_data

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

    return amount_clean_data, amount_clean_intervals_2sec, clean_interval_lengths, amount_annotated_data, amount_annotated_overlap

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
            hovertemplate='Length-range (in seconds)=%{x}, Amount of intervals=%{y}' + '<extra></extra>',
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

def get_power_spectrum_plot(f, Pxx_den):
    """Generates power spectrum plot.

    Args:
        f (array): Sample frequencies.
        Pxx_den (array): Power densities.

    Returns:
        plotly.graph_objs.Figure: Plot of power spectrum.
    """
    fig = Figure()

    fig.add_trace(
        Scattergl(
            x=f,
            y=Pxx_den,
            hovertemplate='Frequency (in Hz)=%{x:.2f}, Power density=%{y:.2f}' + '<extra></extra>',
            marker=dict(color='#4796c5')
        )
    )

    fig.update_layout(
        xaxis=dict(
            title_text='Frequencies (in Hz)',
        ),
        yaxis=dict(
            title_text='Power density'
        ),
    )

    return fig

def calc_power_spectrum(sample_rate, selected_data):
    """Calculate power spectrum of selected data using SciPy's welch method.

    Args:
        sample_rate (float): Sample rate of selected data.
        selected_data (array): Selected data to calculate power spectrum of.

    Returns:
        tuple(array, array): Tuple of sample frequencies and corresponding power densities.
    """
    f, Pxx_den = welch(selected_data, sample_rate)

    return f, Pxx_den

def get_most_prominent_freq(f, Pxx_den):
    """Calculate most prominent frequency of selected data.

    Args:
        f (array): Sample frequencies.
        Pxx_den (array): Power densities.

    Returns:
        float: Frequency with highest density.
    """
    temp_list = Pxx_den.tolist()
    maximum_peak = temp_list.index(Pxx_den.max())
    maximum_peak_value = f[maximum_peak]
    # print(maximum_peak_value)

    return maximum_peak_value
