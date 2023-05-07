import numpy as np
from scipy.signal import welch

from plotly.graph_objs import Figure, Scattergl

from mne.viz.topomap import _get_pos_outlines

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.bad_channel_helperfunctions import get_bad_channels
import constants as c
import globals


def get_channel_locations_plot(raw):
    """Generates channel-topography plot of given raw object.

    Args:
        raw (mne.io.Raw): Raw object to get channel locations from.

    Returns:
        plotly.graph_objs.Figure: Plot of channel-topography.
    """
    # Get channel locations through MNE
    if raw.info['dig']:
        pos, outlines = _get_pos_outlines(raw.info, np.arange(len(raw.info['chs'])), 'auto')
        
        head_coordinates = outlines['mask_pos']
        head_markers = []
        head_markers.append(min(head_coordinates[0]))
        head_markers.append(max(head_coordinates[0]))
        head_markers.append(min(head_coordinates[1]))
        head_markers.append(max(head_coordinates[1]))
        # print(head_markers)
    else:
        pos = np.array([])
        outlines = []
        head_markers = []
    # print(pos)
    # print(outlines)
    
    chs = raw.info['chs']
    channel_coordinates = pos  # np.empty((len(chs), 2))  # manual
    channel_names = []

    for index, channel in enumerate(chs):
        # channel_coordinates[index] = channel['loc'][:2]  # manual
        channel_names.append(channel['ch_name'])
        
    bad_channels = raw.info['bads']

    # Optional to scale channel locations
    # channel_coordinates = channel_coordinates * 1000

    topography_plot = Figure()

    if channel_coordinates.size > 0:
        for channel_index, channel in enumerate(channel_names):
            topography_plot.add_trace(
                Scattergl(
                    x=[channel_coordinates[channel_index, 0]],
                    y=[channel_coordinates[channel_index, 1]],
                    customdata=[channel],
                    mode="markers+text",
                    name=channel,
                    text=channel,
                    textposition="bottom center" if channel_coordinates[channel_index, 1] <= 0 else 'top center',
                    hovertemplate='<b>%{fullData.name}</b>' + '<extra></extra>' if channel not in bad_channels else '<b> Bad channel | %{fullData.name}</b>' + '<extra></extra>',
                    marker=dict(color='black') if channel not in bad_channels else dict(color='red')
                )
            )
        
        if head_markers:
            topography_plot.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=head_markers[0],
                x1=head_markers[1],
                y0=head_markers[2],
                y1=head_markers[3],
                line_color="black",
            )

    topography_plot.update_layout(
        dragmode='select',
        showlegend=False,
        clickmode='event+select',
        # plot_bgcolor='#dfdfdf',
    )
    
    topography_plot.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=False,
    )
    
    topography_plot.update_xaxes(
        showgrid=False,
    )

    # topography_plot.update_xaxes(
    #     # title_text='Time (in seconds)'
    #     # showgrid=True,
    #     # zeroline=False,
    #     # constrain='domain',
    #     # range=(-0.2, 10.2),  # Start x-axis range to show approx. 10 seconds
    # )

    return topography_plot

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

def preprocess_EEG(raw, high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation):
    # Bandpass-filter
    if (high_pass or low_pass) and not (float(high_pass) == globals.raw.info['highpass'] and float(low_pass) == globals.raw.info['lowpass']):
        # print(high_pass)
        # print(low_pass)
        print('Applying bandpass-filter')
        raw.filter(high_pass, low_pass, method='fir', fir_window='blackman')

    print(raw.info['bads'])

    # Bad-channel detection
    if bad_channel_detection == 'None':
        print('No automatic bad-channel detection')
        bad_channel_detection = None
    elif bad_channel_detection == 'AutoReject':
        print('Automatic bad-channel detection using AutoReject')
        bad_channel_detection = 'AutoReject'
    elif bad_channel_detection == 'RANSAC':
        print('Automatic bad-channel detection using RANSAC')
        bad_channel_detection = 'RANSAC'

    if bad_channel_detection:
        print('Performing automatic bad channel detection')
        detected_bad_channels = get_bad_channels(globals.raw, bad_channel_detection)
        # print(detected_bad_channels)

        total_bad_channels = globals.raw.info['bads']
        for bad_channel in detected_bad_channels:
            if bad_channel not in total_bad_channels:
                total_bad_channels.append(bad_channel)

        raw.info['bads'] = total_bad_channels

    # Re-referencing
    if reference:
        # print('Reference: {}'.format(reference))
        if reference == 'None':
            print('No re-referencing')
            reference = None
        elif reference != 'average':
            reference = [reference]

        if reference:
            print('Applying custom reference {}'.format(reference))
            raw.set_eeg_reference(reference)

    # Bad-channel interpolation
    if bad_channel_interpolation:
        # print(globals.raw.info['bads'])
        print('Performing bad-channel interpolation')
        raw = raw.interpolate_bads(reset_bads=False)
        
    return raw

def _get_scaling(EEG_scale):
    """Calculates scaling factor to multiply data with for given scale.

    Args:
        EEG_scale (float): Desired scale of data. Defaults to 1e-6.

    Returns:
        float: Scaling factor to multiply data with.
    """
    if EEG_scale:
        temp = EEG_scale
        temp_string = str(EEG_scale)

        if 'e' in temp_string:
            e_index = temp_string.index('e')
            zeros = int(temp_string[e_index + 2:])
            temp *= 10 ** zeros
            temp *= 10 ** zeros
        else:
            zeros = 0
            while temp < 1:
                temp *= 10
                zeros += 1
            temp *= 10 ** zeros

        scaling_factor = temp

        # print('Apply custom scaling {}'.format(scaling_factor))
    else:
        scaling_factor = c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS
        
    return scaling_factor
        
def _get_offset(channel_offset):
    """Returns offset factor to multiply data with.

    Args:
        channel_offset (int): Offset between channels. Defaults to 40 (μV).

    Returns:
        int: Offset factor to multiply data with.
    """
    if channel_offset or channel_offset == 0:
        offset_factor = channel_offset
    else:
        offset_factor = c.DEFAULT_Y_AXIS_OFFSET
        
    return offset_factor

def _get_time(EEG_data, sample_frequency):
    """Calculates timescale and recording length of given EEG data.

    Args:
        EEG_data (array): EEG data.
        sample_frequency (float): Sample frequency of data.

    Returns:
        tuple(array, float): Array of all timepoints, recording length.
    """
    # Create time-scale for x-axis of visualization
    timestep = 1 / sample_frequency

    timescale = [timestep * i for i in range(EEG_data.shape[0])]

    recording_length = float(timescale[-1] + timestep)
    print('Length of recording in seconds: {}'.format(recording_length))
    
    return timescale, recording_length

def _get_plotting_data(raw, file_name, selected_channel_names, EEG_scale, channel_offset, model_output=[], model_channels=[]):
    """Generates dict holding all relevant data from raw object and model outputs for plotting.

    Args:
        raw (mne.io.Raw): Raw object to get data from.
        file_name (string): File-name.
        selected_channel_names (list): List of strings of selected channel names to plot.
        EEG_scale (float): Desired scaling.
        channel_offset (int): Desired channel offset.
        model_output (list, optional): List of arrays of model outputs. Defaults to [].
        model_channels (list, optional): List of lists of strings of channel names model output is based on. Defaults to [].

    Returns:
        dict: Dict holding all relevant data from raw object and model outputs for plotting.
    """
    plotting_data = {'EEG': {}, 'model': [], 'plot': {}}
    
    plotting_data['EEG']['file_name'] = file_name

    plotting_data['EEG']['scaling_factor'] = _get_scaling(EEG_scale)
    plotting_data['plot']['offset_factor'] = _get_offset(channel_offset)

    if selected_channel_names:
        check = all(channel in raw.ch_names for channel in selected_channel_names)
        if check:
            plotting_data['EEG']['EEG_data'] = np.transpose(raw.get_data(selected_channel_names))
            plotting_data['EEG']['channel_names'] = selected_channel_names
    else:
        print('Displaying all channels')
        plotting_data['EEG']['EEG_data'] = np.transpose(raw.get_data())
        plotting_data['EEG']['channel_names'] = raw.ch_names

    plotting_data['EEG']['timescale'], plotting_data['EEG']['recording_length'] = _get_time(plotting_data['EEG']['EEG_data'], raw.info['sfreq'])

    # Calculate offset for y-axis
    offset_EEG = plotting_data['EEG']['EEG_data'].copy() 
    offset_EEG = offset_EEG * plotting_data['EEG']['scaling_factor']
    
    default_channel_colors = []
    highlighted_channel_colors = []
    channel_visibility = []

    for channel_index in range(offset_EEG.shape[1]):
        # Calculate offset for y-axis
        offset_EEG[:, channel_index] = offset_EEG[:, channel_index] + ((plotting_data['plot']['offset_factor']) * (len(plotting_data['EEG']['channel_names']) - 1 - channel_index))  # First channel goes to top of the plot
        
        # Channel colors
        if plotting_data['EEG']['channel_names'][channel_index] in raw.info['bads']:
            default_channel_colors.append(c.BAD_CHANNEL_COLOR)
            channel_visibility.append(False)
        else:
            default_channel_colors.append('black')
            channel_visibility.append(True)

        model_channel = False
        for model in range(len(model_channels)):
            if model_channels[model]:
                if plotting_data['EEG']['channel_names'][channel_index] in model_channels[model]:
                    model_channel = True
            
        if model_channel:
            highlighted_channel_colors.append('blue')
        elif plotting_data['EEG']['channel_names'][channel_index] in raw.info['bads']:
            highlighted_channel_colors.append(c.BAD_CHANNEL_COLOR)
        else:
            highlighted_channel_colors.append('black')
            
    plotting_data['EEG']['offset_EEG_data'] = offset_EEG
            
    for model_index, model_array in enumerate(model_output):
        default_channel_colors.append(model_array)
        highlighted_channel_colors.append(model_array)
        channel_visibility.append(True)
        
        plotting_data['model'].append({})
        plotting_data['model'][model_index]['model_data'] = model_array
        plotting_data['model'][model_index]['model_channels'] = model_channels[model_index]

        plotting_data['model'][model_index]['model_timescale'] = np.linspace(0, plotting_data['EEG']['recording_length'], num=model_array.shape[0])

        plotting_data['model'][model_index]['offset_model_data'] = [-((2 + model_index) * (plotting_data['plot']['offset_factor'])) for i in range(len(plotting_data['model'][model_index]['model_timescale']))]
    
    plotting_data['EEG']['default_channel_colors'] = default_channel_colors
    plotting_data['EEG']['highlighted_channel_colors'] = highlighted_channel_colors
    plotting_data['EEG']['channel_visibility'] = channel_visibility
    
    # y_ticks_model_output = np.arange((-len(model_output) - 1), -1)
    # y_ticks_channels = np.arange(0, len(plotting_data['EEG']['channel_names']))
    # y_ticks = np.concatenate((y_ticks_model_output, y_ticks_channels))
    # y_ticks = y_ticks * (plotting_data['plot']['offset_factor'])
    
    # plotting_data['plot']['y_ticks'] = y_ticks

    # y_tick_labels = [channel_name for channel_name in plotting_data['EEG']['channel_names']]
    # for model_id in range(len(model_output)):
    #     y_tick_labels.append('M{}'.format(model_id))
    # y_tick_labels.reverse()
    
    # plotting_data['plot']['y_tick_labels'] = y_tick_labels.copy()
            
    return plotting_data

def get_EEG_figure(file_name, raw, selected_channel_names, annotation_label, EEG_scale=None, channel_offset=None, model_output=None, model_channels=[], use_slider=False, show_annotations_only=False):
    """Generates initial EEG figure.

    Args:
        file_name (string): File name.
        raw (mne.io.Raw): Raw object to plot data from.
        selected_channel_names (list): List of strings of selected channel names to plot.
        annotation_label (string); Label for new annotations.
        EEG_scale (float): Desired scaling.
        channel_offset (int): Desired channel offset.
        model_output (list, optional): List of arrays of model outputs. Defaults to [].
        model_channels (list, optional): List of lists of strings of channel names model output is based on. Defaults to [].
        use_slider (bool, optional): Whether or not to activate view-slider. Defaults to False.

    Returns:
        plotly.graph_objs.Figure: Plot of EEG data.
    """
    fig = Figure()
    
    plotting_data = _get_plotting_data(raw, file_name, selected_channel_names, EEG_scale, channel_offset, model_output, model_channels)
    globals.plotting_data = plotting_data.copy()    
    
    fig = get_EEG_plot(plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

    return fig

def get_EEG_plot(data_to_plot, x0, x1, annotation_label, use_slider=False, show_annotations_only=False):
    """Generates EEG plots.

    Args:
        plotting_data (dict): Dict holding all relevant data from raw object and model outputs for plotting.
        x0 (float): X-coordinate (in seconds) to start plot.
        x1 (float): X-coordinate (in seconds) to end plot.
        annotation_label (string); Label for new annotations.
        use_slider (bool, optional): Whether or not to activate view-slider. Defaults to False.

    Returns:
        plotly.graph_objs.Figure: Plot of EEG data.
    """
    fig = Figure()
    
    plotting_data = data_to_plot.copy()

    index_0 = None
    index_1 = None
    
    for index, timepoint in enumerate(plotting_data['EEG']['timescale']):
        if timepoint > x0:
            index_0 = index
            break
        
    for index, timepoint in enumerate(plotting_data['EEG']['timescale']):
        if timepoint > x1:
            index_1 = index
            break
    
    for channel_index in range(plotting_data['EEG']['offset_EEG_data'].shape[1]):      
        fig.add_trace(
            Scattergl(
                x=plotting_data['EEG']['timescale'][index_0:index_1],
                y=plotting_data['EEG']['offset_EEG_data'][index_0:index_1, channel_index],
                marker=dict(color=plotting_data['EEG']['default_channel_colors'][channel_index], size=0.1),
                name=plotting_data['EEG']['channel_names'][channel_index],
                customdata=plotting_data['EEG']['EEG_data'][index_0:index_1, channel_index] * plotting_data['EEG']['scaling_factor'],  # y-data without offset
                hovertemplate='<b>%{fullData.name}</b> | Time (in seconds)=%{x:.2f}, Amplitude (in μV)=%{customdata:.3f}' + '<extra></extra>' if plotting_data['EEG']['scaling_factor'] == c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS else '<b>%{fullData.name}</b> | Time (in seconds)=%{x:.2f}, Amplitude (scaled)=%{customdata:.3f}' + '<extra></extra>',
                mode='lines+markers'
            )
        )

    default_channel_colors = plotting_data['EEG']['default_channel_colors'].copy()
    highlighted_channel_colors = plotting_data['EEG']['highlighted_channel_colors'].copy()

    model_index_0 = None
    model_index_1 = None

    # Model predictions
    for model_index in range(len(plotting_data['model'])):
        for index, timepoint in enumerate(plotting_data['model'][model_index]['model_timescale']):
            if timepoint > x0:
                model_index_0 = index
                break
            
        for index, timepoint in enumerate(plotting_data['model'][model_index]['model_timescale']):
            if timepoint > x1:
                model_index_1 = index
                break

        trace_number = plotting_data['EEG']['offset_EEG_data'].shape[1] + model_index

        default_channel_colors[trace_number] = plotting_data['EEG']['default_channel_colors'][trace_number][model_index_0:model_index_1]
        highlighted_channel_colors[trace_number] = plotting_data['EEG']['highlighted_channel_colors'][trace_number][model_index_0:model_index_1]

        fig.add_trace(
            Scattergl(
                x=plotting_data['model'][model_index]['model_timescale'][model_index_0:model_index_1],
                y=plotting_data['model'][model_index]['offset_model_data'][model_index_0:model_index_1],
                marker=dict(
                    size=10,
                    cmax=1,
                    cmin=0,
                    color=plotting_data['model'][model_index]['model_data'][model_index_0:model_index_1],
                    # colorbar=dict(
                    #     title='Colorbar',
                    #     yanchor="top", 
                    #     y=1, 
                    #     x=1
                    # ),
                    colorscale='RdBu_r'
                ),
                name='M{}'.format(model_index),
                mode='markers',
                customdata=plotting_data['model'][model_index]['model_data'][model_index_0:model_index_1],
                hovertemplate='Time=%{x:.2f}, Prediction=%{customdata:.2f}<extra><b>%{fullData.name}</b></extra>'
            )
        )
    
    longest_channel_name_length = len(max(plotting_data['EEG']['channel_names'], key=len))

    fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#dfdfdf',
        title=dict(
            text=plotting_data['EEG']['file_name'],
            y=0.98,
            x=0.5 if not plotting_data['model'] else 0.6,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            # title_text='Time (in seconds)',
            rangeslider=dict(
                visible=True if use_slider else False,
                thickness=0.04
            ),
            type="linear",
        ),

        legend=dict(
            # itemwidth=30
            # orientation="h",
            # yanchor="bottom",
            # y=1,
            # xanchor="left",
            x=1.01
        ),

        autosize=False,
        margin=dict(
            autoexpand=False,
            l=longest_channel_name_length * 6 + 15,  #30
            r=longest_channel_name_length * 6 + 80,  #115
            # b=0,
            t=50,
            pad=5,
        ),

        dragmode='drawrect',
        newshape=dict(
            fillcolor=globals.annotation_label_colors[annotation_label],
            opacity=0.6,
            drawdirection='vertical',
            layer='below',
            line_width=0
        ),
    )

    y_ticks_model_output = np.arange((-len(plotting_data['model']) - 1), -1)
    y_ticks_channels = np.arange(0, len(plotting_data['EEG']['channel_names']))
    y_ticks = np.concatenate((y_ticks_model_output, y_ticks_channels))
    y_ticks = y_ticks * (plotting_data['plot']['offset_factor'])
    
    plotting_data['plot']['y_ticks'] = y_ticks

    y_tick_labels = [channel_name for channel_name in plotting_data['EEG']['channel_names']]
    for model_id in range(len(plotting_data['model'])):
        y_tick_labels.append('M{}'.format(model_id))
    y_tick_labels.reverse()
    
    plotting_data['plot']['y_tick_labels'] = y_tick_labels

    fig.update_yaxes(
        tickmode='array',
        tickvals=plotting_data['plot']['y_ticks'],
        ticktext=plotting_data['plot']['y_tick_labels'],
        showgrid=False,
        zeroline=False,
        fixedrange=False,
        range=((-(2 + len(plotting_data['model'])) * (c.DEFAULT_Y_AXIS_OFFSET)), ((len(plotting_data['EEG']['channel_names']) + 1) * (c.DEFAULT_Y_AXIS_OFFSET)))  # Start y-axis range to cut off peaks
    )
    fig.update_xaxes(
        # title_text='Time (in seconds)'
        showgrid=True,
        zeroline=False,
        constrain='domain',
        range=(x0, x1) if (not use_slider or show_annotations_only) else (x0, x0 + 11),
    )

    # Add annotations
    marked_annotations = get_annotations(globals.raw)
    print(marked_annotations)
    
    for annotation in marked_annotations:
        # if not ((annotation[0] < globals.x0 and annotation[1] < globals.x0) or (annotation[0] > globals.x1 and annotation[1] > globals.x1)):
        fig.add_vrect(
            editable=True,
            x0=annotation[0],
            x1=annotation[1],
            # annotation_text=annotation[2],
            fillcolor=globals.annotation_label_colors[annotation[2]] if annotation[2] in globals.annotation_label_colors.keys() else 'red',
            opacity=0.6,
            layer='below',
            line_width=0
        )
   
    fig.update_layout(
        updatemenus=list([
            dict(
                buttons=list([
                    dict(label="Reset time-axis",  
                        method="relayout", 
                        args=[{
                            "xaxis.range[0]": x0,
                            "xaxis.range[1]": x1 if (not use_slider) or show_annotations_only else x0 + 11,
                        }]
                    ),
                    dict(label="Reset channel-axis",  
                        method="relayout", 
                        args=[{
                            "yaxis.range[0]": (-(2 + len(plotting_data['model'])) * (c.DEFAULT_Y_AXIS_OFFSET)),
                            "yaxis.range[1]": ((len(plotting_data['EEG']['channel_names']) + 1) * (c.DEFAULT_Y_AXIS_OFFSET))
                        }]
                    ),
                    dict(label='Hide/show bad channels',
                        method='restyle',
                        args2=[{'visible': True}],
                        args=[{'visible': plotting_data['EEG']['channel_visibility']}]
                    ),
                    dict(label='Highlight model-channels',
                        method='restyle',
                        args2=[{'marker.color': default_channel_colors}],
                        args=[{'marker.color': highlighted_channel_colors}],
                        visible=True if plotting_data['model'] else False
                    ),
                ]),
                direction = 'left',
                # active=4,
                # pad = {'l': 100, 't': 10},
                showactive = False,
                type = 'buttons',
                xanchor = 'left',
                yanchor = 'top',
                x = 0,
                y = 1.08,
                bgcolor = '#fafafa',
                bordercolor = '#c7c7c7'
            )
        ])
    )

    return fig