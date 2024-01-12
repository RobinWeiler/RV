import time

import numpy as np

import mne

from dash import Patch
from plotly.graph_objs import Figure, Scattergl, Scatter

import constants as c
import globals


def _channel_name_sorting_key(channel_name):
    if channel_name.startswith('E'):
        return int(channel_name[1:])  # sort by the numeric part
    else:
        return float('inf')  # 'Cz' or any other string not starting with 'E' goes to the end

def _get_list_for_displaying(example_list):
    if example_list:
        return [element + ', ' if element_index != len(example_list) - 1 else element for element_index, element in enumerate(example_list)]
    else:
        return []

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

# def _get_time(EEG_data, sample_frequency):
#     """Calculates timescale and recording length of given EEG data.

#     Args:
#         EEG_data (array): EEG data.
#         sample_frequency (float): Sample frequency of data.

#     Returns:
#         tuple(array, float): Array of all timepoints, recording length.
#     """
#     # Create time-scale for x-axis of visualization
#     timestep = 1 / sample_frequency

#     timescale = [timestep * i for i in range(EEG_data.shape[0])]

#     recording_length = float(timescale[-1] + timestep)
#     print('Length of recording in seconds: {}'.format(recording_length))
    
#     return timescale, recording_length

def _get_plotting_data(raw, file_name, selected_channel_names, EEG_scale, channel_offset, model_output=[], model_channels=[], reorder_channels=False):
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
            # plotting_data['EEG']['EEG_data'] = np.transpose(raw.get_data(selected_channel_names))
            plotting_data['EEG']['channel_names'] = selected_channel_names
    else:
        print('Displaying all channels')
        # plotting_data['EEG']['EEG_data'] = np.transpose(raw.get_data())
        plotting_data['EEG']['channel_names'] = raw.ch_names
        
        if reorder_channels:
            channel_order = []
            for region in c.CHANNEL_TO_REGION_128.keys():
                for channel in c.CHANNEL_TO_REGION_128[region]:
                    channel_name = 'E{}'.format(channel)
                    if channel_name in plotting_data['EEG']['channel_names']:
                        channel_order.append(channel_name)
            channel_order.append('Cz')

            globals.raw.reorder_channels(channel_order)
            raw.reorder_channels(channel_order)
            plotting_data['EEG']['channel_names'] = globals.raw.ch_names

    # plotting_data['EEG']['timescale'], plotting_data['EEG']['recording_length'] = _get_time(plotting_data['EEG']['EEG_data'], raw.info['sfreq'])
    plotting_data['EEG']['recording_length'] = len(raw) / raw.info['sfreq']

    # Calculate offset for y-axis
    # offset_EEG = plotting_data['EEG']['EEG_data'].copy() 
    # offset_EEG = offset_EEG * plotting_data['EEG']['scaling_factor']

    eog_channels_indices = mne.pick_types(raw.info, eog=True)
    eog_channels = []
    for channel_index in eog_channels_indices:
        eog_channels.append(raw.ch_names[channel_index])
    # print(eog_channels)

    plotting_data['EEG']['eog_channels'] = eog_channels

    # plotting_data['EEG']['offset_EEG_data'] = offset_EEG

    for model_index, model_array in enumerate(model_output):
        plotting_data['model'].append({})
        plotting_data['model'][model_index]['model_data'] = model_array
        plotting_data['model'][model_index]['model_channels'] = model_channels[model_index]

        plotting_data['model'][model_index]['model_timescale'] = np.linspace(0, plotting_data['EEG']['recording_length'], num=model_array.shape[0])

        plotting_data['model'][model_index]['offset_model_data'] = [-((2 + model_index) * (plotting_data['plot']['offset_factor'])) for i in range(len(plotting_data['model'][model_index]['model_timescale']))]

    # y_tick_labels = [channel_name for channel_name in plotting_data['EEG']['channel_names']]
    # for model_id in range(len(plotting_data['model'])):
    #     y_tick_labels.append('M{}'.format(model_id))
    # # y_tick_labels.reverse()

    # plotting_data['plot']['y_tick_labels'] = y_tick_labels

    return plotting_data

def _get_next_segment(raw, x0, x1, channels, scaling_factor, offset_factor, skip_hoverinfo=False, use_slider=False, show_annotations_only=False, reorder_channels=False):
    patched_fig = Patch()
    
    # Get data of new segment
    index_0 = raw.time_as_index(x0)[0] if x0 > 0 else 0
    index_1 = raw.time_as_index(x1)[0]

    data_subset, times_subset = raw[channels, index_0:index_1]
    data_subset = data_subset * scaling_factor

    if not skip_hoverinfo:
        custom_data = data_subset.copy()

    # Update EEG traces
    y_ticks_model_output = np.arange((-len(globals.plotting_data['model']) - 1), -1)
    y_ticks_channels = np.arange(0, len(globals.plotting_data['EEG']['channel_names']))
    y_ticks = np.concatenate((y_ticks_model_output, y_ticks_channels))
    y_ticks = y_ticks * (globals.plotting_data['plot']['offset_factor'])
    
    if reorder_channels:
        region_offset = np.zeros(len(globals.plotting_data['EEG']['channel_names']), dtype=np.int64)

        region_names = list(c.CHANNEL_TO_REGION_128.keys())
        region_names.reverse()
        counter = 1  # Cz in position 0

        for index, region in enumerate(region_names):
            for _ in range(len(c.CHANNEL_TO_REGION_128[region])):
                if counter < len(globals.plotting_data['EEG']['channel_names']):
                    region_offset[counter] = index * globals.plotting_data['plot']['offset_factor'] * 2
                    counter += 1

        # region_offset = np.flip(region_offset)

        if len(globals.plotting_data['model']) > 0:
            y_ticks[len(globals.plotting_data['model']):] += region_offset
        else:
            y_ticks += region_offset
    y_ticks = np.flip(y_ticks)

    if len(globals.plotting_data['model']) > 0:
        data_subset += y_ticks.reshape(-1, 1)[:-len(globals.plotting_data['model'])]
    else:
        data_subset += y_ticks.reshape(-1, 1)

    for channel_index in range(len(channels)):
        # data_subset[channel_index, :] = data_subset[channel_index, :] + (offset_factor * (len(channels) - 1 - channel_index))  # First channel goes to top of the plot

        patched_fig['data'][channel_index]['x'] = times_subset
        patched_fig['data'][channel_index]['y'] = data_subset[channel_index, :]

        if not skip_hoverinfo:
            patched_fig['data'][channel_index]['customdata'] = custom_data[channel_index]

    # Update model predictions
    for model_index in range(len(globals.plotting_data['model'])):
        for index, timepoint in enumerate(globals.plotting_data['model'][model_index]['model_timescale']):
            if timepoint >= x0:
                model_index_0 = index
                break
            
        for index, timepoint in enumerate(globals.plotting_data['model'][model_index]['model_timescale']):
            if timepoint >= x1:
                model_index_1 = index
                break
            else:
                model_index_1 = len(globals.plotting_data['model'][model_index]['model_timescale']) - 1

        patched_fig['data'][len(channels) + model_index]['x'] = globals.plotting_data['model'][model_index]['model_timescale'][model_index_0:model_index_1]
        patched_fig['data'][len(channels) + model_index]['y'] = globals.plotting_data['model'][model_index]['offset_model_data'][model_index_0:model_index_1]
        patched_fig['data'][len(channels) + model_index]['marker']['color'] = globals.plotting_data['model'][model_index]['model_data'][model_index_0:model_index_1]

        if not skip_hoverinfo:
            patched_fig['data'][len(channels) + model_index]['customdata'] = globals.plotting_data['model'][model_index]['model_data'][model_index_0:model_index_1]

    visible_annotations = [annotation for annotation in globals.marked_annotations if globals.annotation_label_colors[annotation[2]] != 'hide']

    patched_fig['layout']['xaxis']['range'] = (x0, x1) if (not use_slider or (show_annotations_only and len(visible_annotations) > 0)) else (x0, x0 + 11)

    patched_fig['layout']['updatemenus'][0]['buttons'][0]['args'][0]['xaxis.range[0]'] = x0
    patched_fig['layout']['updatemenus'][0]['buttons'][0]['args'][0]['xaxis.range[1]'] = x1 if (not use_slider or (show_annotations_only and len(visible_annotations) > 0)) else x0 + 11
        
    return patched_fig

def get_EEG_figure(file_name, raw, selected_channel_names, annotation_label, show_annotation_labels=False, EEG_scale=None, channel_offset=None, model_output=None, model_channels=[], use_slider=False, show_annotations_only=False, skip_hoverinfo=False, hide_bad_channels=False, highlight_model_channels=False, reorder_channels=False):
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
    
    globals.plotting_data = _get_plotting_data(raw, file_name, selected_channel_names, EEG_scale, channel_offset, model_output, model_channels, reorder_channels)
    # globals.plotting_data = plotting_data.copy()    
    
    fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, hide_bad_channels, highlight_model_channels, reorder_channels)

    return fig

def get_EEG_plot(plotting_data, x0, x1, annotation_label, show_annotation_labels=False, use_slider=False, show_annotations_only=False, skip_hoverinfo=False, hide_bad_channels=False, highlight_model_channels=False, reorder_channels=False):
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

    index_0 = globals.viewing_raw.time_as_index(x0)[0] if x0 > 0 else 0
    index_1 = globals.viewing_raw.time_as_index(x1)[0]

    data_subset, times_subset = globals.viewing_raw[plotting_data['EEG']['channel_names'], index_0:index_1]
    data_subset = data_subset * plotting_data['EEG']['scaling_factor']

    if not skip_hoverinfo:
        custom_data = data_subset.copy()

    # for channel_index in range(len(plotting_data['EEG']['channel_names'])):
    #     # data_subset[channel_index, :] = data_subset[channel_index, :] + ((plotting_data['plot']['offset_factor']) * (len(plotting_data['EEG']['channel_names']) - 1 - channel_index))  # First channel goes to top of the plot
    #     data_subset[channel_index, :] = data_subset[channel_index, :] + plotting_data['plot']['y_ticks'][channel_index]

    t1 = time.time()

    y_ticks_model_output = np.arange((-len(plotting_data['model']) - 1), -1)
    y_ticks_channels = np.arange(0, len(plotting_data['EEG']['channel_names']))
    y_ticks = np.concatenate((y_ticks_model_output, y_ticks_channels))
    y_ticks = y_ticks * (plotting_data['plot']['offset_factor'])
    
    if reorder_channels:
        region_offset = np.zeros(len(globals.plotting_data['EEG']['channel_names']), dtype=np.int64)

        region_names = list(c.CHANNEL_TO_REGION_128.keys())
        region_names.reverse()
        counter = 1  # Cz in position 0

        for index, region in enumerate(region_names):
            for _ in range(len(c.CHANNEL_TO_REGION_128[region])):
                if counter < len(globals.plotting_data['EEG']['channel_names']):
                    region_offset[counter] = index * globals.plotting_data['plot']['offset_factor'] * 2
                    counter += 1

        # region_offset = np.flip(region_offset)

        if len(plotting_data['model']) > 0:
            y_ticks[len(plotting_data['model']):] += region_offset
        else:
            y_ticks += region_offset
    y_ticks = np.flip(y_ticks)

    if len(plotting_data['model']) > 0:
        data_subset += y_ticks.reshape(-1, 1)[:-len(plotting_data['model'])]
    else:
        data_subset += y_ticks.reshape(-1, 1)

    for channel_index in range(data_subset.shape[0]):   
        channel_name = plotting_data['EEG']['channel_names'][channel_index]

        channel_color = 'black'
        if channel_name in plotting_data['EEG']['eog_channels']:
            channel_color = 'blue'
        if highlight_model_channels:
            for model_index in range(len(plotting_data['model'])):
                if channel_name in plotting_data['model'][model_index]['model_channels']:
                    channel_color = c.MODEL_CHANNEL_COLOR
                    break
        if channel_name in globals.raw.info['bads']:
            if channel_name in globals.disagreed_bad_channels:
                channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
            else:
                channel_color = c.BAD_CHANNEL_COLOR

        region_name = None
        if reorder_channels and not channel_name == 'Cz':
            for region, channels in c.CHANNEL_TO_REGION_128.items():
                if int(channel_name[1:]) in channels:
                    region_name = region
                    break

        fig.add_trace(
            Scattergl(
                x=times_subset,  # plotting_data['EEG']['timescale'][index_0:index_1],
                y=data_subset[channel_index, :],  # plotting_data['EEG']['offset_EEG_data'][index_0:index_1, channel_index],
                marker=dict(color=channel_color, size=0.1),
                name=channel_name if not region_name else channel_name + ' ' + region_name,
                customdata=custom_data[channel_index] if not skip_hoverinfo else None,  # plotting_data['EEG']['EEG_data'][index_0:index_1, channel_index] * plotting_data['EEG']['scaling_factor'] if not skip_hoverinfo else None,  # y-data without offset
                hoverinfo='none' if skip_hoverinfo else 'all',
                hovertemplate='' if skip_hoverinfo else '<b>%{fullData.name}</b> | Time = %{x:.2f} seconds, Amplitude = %{customdata:.2f} μV' + '<extra></extra>' if plotting_data['EEG']['scaling_factor'] == c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS else '<b>%{fullData.name}</b> | Time = %{x:.2f} seconds, Amplitude (scaled) = %{customdata:.2f}' + '<extra></extra>',
                mode='lines+markers',
                visible=True if channel_name not in globals.raw.info['bads'] or not hide_bad_channels else False
            )
        )
    t2 = time.time()
    print(t2-t1)

    model_index_0 = None
    model_index_1 = None

    # Model predictions
    for model_index in range(len(plotting_data['model'])):
        for index, timepoint in enumerate(plotting_data['model'][model_index]['model_timescale']):
            if timepoint >= x0:
                model_index_0 = index
                break
            
        for index, timepoint in enumerate(plotting_data['model'][model_index]['model_timescale']):
            if timepoint >= x1:
                model_index_1 = index
                break
            else:
                model_index_1 = len(globals.plotting_data['model'][model_index]['model_timescale']) - 1

        # trace_number = data_subset.shape[0] + model_index

        # default_channel_colors[trace_number] = plotting_data['EEG']['default_channel_colors'][trace_number][model_index_0:model_index_1]
        # highlighted_channel_colors[trace_number] = plotting_data['EEG']['highlighted_channel_colors'][trace_number][model_index_0:model_index_1]

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
                customdata=plotting_data['model'][model_index]['model_data'][model_index_0:model_index_1] if not skip_hoverinfo else None,
                hoverinfo='none' if skip_hoverinfo else 'all',
                hovertemplate='' if skip_hoverinfo else 'Time=%{x:.2f}, Prediction=%{customdata:.2f}<extra><b>%{fullData.name}</b></extra>'
            )
        )
    
    longest_channel_name_length = len(max(plotting_data['EEG']['channel_names'], key=len)) 
    if reorder_channels:
        longest_region_name_length = len(max(c.CHANNEL_TO_REGION_128.keys())) 

    fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#dfdfdf',
        title=dict(
            text=plotting_data['EEG']['file_name'],
            y=0.98,
            x=0.5,  # if not plotting_data['model'] else 0.6,
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
            r=(longest_channel_name_length + longest_region_name_length if reorder_channels else longest_channel_name_length) * 6 + 80,  #115
            # b=0,
            t=50,
            pad=5,
        ),

        dragmode='drawrect',
        newshape=dict(
            fillcolor=globals.annotation_label_colors[annotation_label] if globals.annotation_label_colors[annotation_label] != 'hide' else 'red',
            opacity=0.6,
            drawdirection='vertical',
            layer='below',
            line_width=0,
            label={'text': annotation_label if show_annotation_labels else '', 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}},
            name=annotation_label,
            visible=True if globals.annotation_label_colors[annotation_label] != 'hide' else False
            # showlegend=True,
            # legend='legend',
            # legendgroup=annotation_label,
            # legendgrouptitle={'text':annotation_label}
        ),
    )

    # Add annotations
    for annotation in globals.marked_annotations:
        # if not ((annotation[0] < globals.x0 and annotation[1] < globals.x0) or (annotation[0] > globals.x1 and annotation[1] > globals.x1)):
        fig.add_vrect(
            editable=True,
            x0=annotation[0],
            x1=annotation[1],
            # annotation_text=annotation[2],
            fillcolor=globals.annotation_label_colors[annotation[2]] if annotation[2] in globals.annotation_label_colors.keys() and globals.annotation_label_colors[annotation[2]] != 'hide' else 'red',
            opacity=0.6,
            layer='below',
            line_width=0,
            name=annotation[2],
            label={'text': annotation[2] if show_annotation_labels else '', 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}},
            visible=True if globals.annotation_label_colors[annotation[2]] != 'hide' else False
        )

        # Could use Scatter traces to create overview over all annotations
        # fig.add_trace(
        #     Scatter(
        #         x=[annotation[0], annotation[1]],  # plotting_data['EEG']['timescale'][index_0:index_1],
        #         y=[-1000, -1000],
        #         marker=dict(color=globals.annotation_label_colors[annotation[2]] if annotation[2] in globals.annotation_label_colors.keys() else 'red', size=1000),
        #         hoverinfo='none',
        #         mode='lines',
        #         visible=True
        #     )
        # )

    visible_annotations = [annotation for annotation in globals.marked_annotations if globals.annotation_label_colors[annotation[2]] != 'hide']

    y_axis_range_0 = -(2 + len(plotting_data['model'])) * c.DEFAULT_Y_AXIS_OFFSET
    y_axis_range_1 = c.DEFAULT_Y_AXIS_OFFSET * len(plotting_data['EEG']['channel_names'])
    if reorder_channels:
        y_axis_range_1 += c.DEFAULT_Y_AXIS_OFFSET * (len(c.CHANNEL_TO_REGION_128) * 2)
    else:
        y_axis_range_1 += c.DEFAULT_Y_AXIS_OFFSET
    if show_annotation_labels:
        y_axis_range_1 += c.DEFAULT_Y_AXIS_OFFSET * 4
    y_axis_range = (y_axis_range_0, y_axis_range_1)

    y_tick_labels = [channel_name for channel_name in plotting_data['EEG']['channel_names']]
    for model_id in range(len(plotting_data['model'])):
        y_tick_labels.append('M{}'.format(model_id))
    # y_tick_labels.reverse()

    fig.update_yaxes(
        tickmode='array',
        tickvals=y_ticks,
        ticktext=y_tick_labels,
        showgrid=False,
        zeroline=False,
        fixedrange=False,
        # range=((-(2 + len(plotting_data['model'])) * (c.DEFAULT_Y_AXIS_OFFSET)), ((len(plotting_data['EEG']['channel_names']) + (5 if show_annotation_labels else 1)) * (c.DEFAULT_Y_AXIS_OFFSET)))  # Start y-axis range to cut off peaks
        # range=((-(2 + len(plotting_data['model'])) * c.DEFAULT_Y_AXIS_OFFSET), (c.DEFAULT_Y_AXIS_OFFSET * (len(plotting_data['EEG']['channel_names']) + (1 if not (reorder_channels and len(globals.plotting_data['EEG']['channel_names']) == 129) else len(c.CHANNEL_TO_REGION_128) * 2))))  # Start y-axis range to cut off peaks
        range=y_axis_range
    )
    fig.update_xaxes(
        # title_text='Time (in seconds)'
        showgrid=True,
        zeroline=False,
        constrain='domain',
        range=(x0, x1) if (not use_slider or (show_annotations_only and len(visible_annotations) > 0)) else (x0, x0 + 11),
    )

    fig.update_layout(
        updatemenus=list([
            dict(
                buttons=list([
                    dict(label="Reset time-axis",  
                        method="relayout", 
                        args=[{
                            "xaxis.range[0]": x0,
                            "xaxis.range[1]": x1 if (not use_slider) or (show_annotations_only and len(visible_annotations) > 0) else x0 + 11,
                        }]
                    ),
                    dict(label="Reset channel-axis",  
                        method="relayout", 
                        args=[{
                            "yaxis.range[0]": (-(2 + len(plotting_data['model'])) * (c.DEFAULT_Y_AXIS_OFFSET)),
                            # "yaxis.range[1]": ((len(plotting_data['EEG']['channel_names']) + (5 if show_annotation_labels else 1)) * (c.DEFAULT_Y_AXIS_OFFSET))
                            # "yaxis.range[1]": (c.DEFAULT_Y_AXIS_OFFSET * (len(plotting_data['EEG']['channel_names']) + (1 if not (reorder_channels and len(globals.plotting_data['EEG']['channel_names']) == 129) else len(c.CHANNEL_TO_REGION_128) * 2)))
                            "yaxis.range[1]": y_axis_range_1
                        }]
                    ),
                    # dict(label='Hide/show bad channels',
                    #     method='restyle',
                    #     args2=[{'visible': plotting_data['EEG']['default_channel_visibility']}],
                    #     args=[{'visible': plotting_data['EEG']['channel_visibility']}]
                    # ),
                    # dict(label='Highlight model-channels',
                    #     method='restyle',
                    #     args2=[{'marker.color': default_channel_colors}],
                    #     args=[{'marker.color': highlighted_channel_colors}],
                    #     visible=True if plotting_data['model'] else False
                    # ),
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
