import math
import collections
import numpy as np

import dash
from dash.dependencies import Input, Output, State

from plotly.graph_objs import Figure
import plotly.express as px
from skimage import io

from helperfunctions.annotation_helperfunctions import merge_intervals, get_annotations, annotations_to_raw, confidence_intervals
from helperfunctions.loading_helperfunctions import parse_data_file, parse_model_output_file, parse_annotation_file
from helperfunctions.visualization_helperfunctions import get_EEG_figure, calc_power_spectrum, get_most_prominent_freq, get_power_spectrum_plot, get_EEG_plot, preprocess_EEG, _get_scaling, _get_offset
from model.run_model import run_model

import constants as c
import globals


def register_visualization_callbacks(app):

    @app.callback(
        [Output('left-button', 'disabled'), Output('right-button', 'disabled')],
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('show-annotations-only', 'value')]
        # prevent_initial_call=True
    )
    def _update_arrow_buttons(fig, segment_size, show_annotations_only):
        """Disables/enables arrow-buttons based on position of current segment. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, bool): Whether or not to disable left-arrow button, whether or not to disable right-arrow button.
        """
        left_disabled = True
        right_disabled = True

        if globals.plotting_data:
            if show_annotations_only:
                if globals.current_plot_index > 0:
                    left_disabled = False
                if globals.current_plot_index + 1 < len(globals.marked_annotations):
                    right_disabled = False
            elif segment_size:
                if globals.x0 == -0.5 and not globals.x1 > globals.plotting_data['EEG']['recording_length']:
                    right_disabled = False
                elif globals.x1 > globals.plotting_data['EEG']['recording_length']:
                    left_disabled = False
                else:
                    left_disabled = False
                    right_disabled = False

        return left_disabled, right_disabled
    
    @app.callback(
        [Output('segment-slider', 'disabled'), Output('segment-slider', 'max'), Output('segment-slider', 'step'), Output('segment-slider', 'marks')],
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('show-annotations-only', 'value')]
        # prevent_initial_call=True
    )
    def _update_segment_slider(fig, segment_size, show_annotations_only):
        """Disables/enables segment-slider. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, bool): Whether or not to disable segment-slider, max value.
        """
        if globals.plotting_data and segment_size:
            if show_annotations_only and len(globals.marked_annotations) > 0:
                num_segments = int(len(globals.marked_annotations) - 1)
                marks = {i: '{}'.format(i) for i in range(num_segments + 1)}
            else:
                num_segments = int(globals.plotting_data['EEG']['recording_length'] // segment_size)
                marks = {i: '{} - {}'.format(i * segment_size, i * segment_size + segment_size) for i in range(num_segments + 1)}

            # print(num_segments)

            return False, num_segments, 1, marks
        else:
            return True, 1, 1, {0: '0', 1: '1'}
    
    @app.callback(
        [Output('EEG-graph', 'figure', allow_duplicate=True), Output('segment-slider', 'value')],
        [Input('segment-slider', 'value'), Input('left-button', 'n_clicks'), Input('right-button', 'n_clicks')],
        [State('segment-size', 'value'), State('show-annotations-only', 'value'), State('use-slider', 'value'), State('annotation-label', 'value'), State('EEG-graph', 'figure')],
        prevent_initial_call=True
    )
    def _use_segment_slider(segment_slider, left_button, right_button, segment_size, show_annotations_only, use_slider, annotation_label, current_fig):
        """Initiates segment-slider based on segment size. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, int, int, int): Whether or not to disable left-arrow button, max value, step size, init value.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if globals.plotting_data and segment_size:
            if 'segment-slider' in trigger:
                print(segment_slider)

                globals.current_plot_index = segment_slider

                if show_annotations_only and len(globals.marked_annotations) > globals.current_plot_index:
                    globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                    globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2
                else:
                    globals.x0 = segment_size * segment_slider if segment_slider != 0 else -0.5
                    globals.x1 = segment_size + (segment_size * segment_slider) + 0.5

            elif 'left-button' in trigger:
                globals.current_plot_index -= 1

                if show_annotations_only and len(globals.marked_annotations) > globals.current_plot_index:
                    globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                    globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2
                elif segment_size:
                    globals.x0 -= segment_size
                    globals.x1 -= segment_size

            elif 'right-button' in trigger:
                globals.current_plot_index += 1
                # print(globals.current_plot_index)

                if show_annotations_only and len(globals.marked_annotations) > globals.current_plot_index:
                    globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                    globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2
                elif segment_size:
                    globals.x0 += segment_size
                    globals.x1 += segment_size

            updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

            return updated_fig, globals.current_plot_index
        else:
            return current_fig, 0

    @app.callback(
        Output('preload-data', 'children'),
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('use-slider', 'value'), State('show-annotations-only', 'value')],
        prevent_initial_call=True
    )
    def _preload_plots(fig, segment_size, use_slider, show_annotations_only):
        """Preloads 1 following segment and adds it to globals.preloaded_plots. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            use_slider (bool): Whether or not to activate view-slider.
        """
        # if globals.plotting_data:
        #     if segment_size:
        #         print('Preloading segments...')
        #         num_segments = math.ceil(globals.plotting_data['EEG']['recording_length'] / segment_size)
        #         # print(num_segments)
                
        #         upper_bound = globals.current_plot_index + 2 if globals.current_plot_index + 2 < num_segments else num_segments
        #         # print(upper_bound)

        #         globals.preloaded_plots[globals.current_plot_index] = fig

        #         for segment_index in range(upper_bound):
        #             if segment_index not in globals.preloaded_plots:
        #                 new_x0 = segment_index * segment_size - 0.5
        #                 new_x1 = segment_size + segment_index * segment_size + 0.5
        #                 globals.preloaded_plots[segment_index] = get_EEG_plot(globals.plotting_data, new_x0, new_x1, use_slider)
                        # print(segment_index)

    # plot callback
    @app.callback(
        Output('EEG-graph', 'figure'),
        [
            Input('plot-button', 'n_clicks'),
            Input('redraw-button', 'n_clicks'),
            Input('EEG-graph', 'clickData'),
            Input("scale", "value"),
            Input("channel-offset", "value"),
            Input('segment-size', 'value'),
            Input('use-slider', 'value'),
            Input('annotation-label', 'value'),
            Input('annotation-label-color', 'value'),
            Input('reset-models', 'n_clicks'),
            Input("run-model", "value"),
            Input("annotate-model", "value"),
            Input("model-threshold", "value"),
            Input('show-annotations-only', 'value'),
        ],
        [
            State('data-file', 'children'),
            State('selected-channels-dropdown', 'value'),
            State("high-pass", "value"), State("low-pass", "value"),
            State('reference-dropdown', 'value'),
            State('bad-channel-detection-dropdown', 'value'), State("bad-channel-interpolation", "value"),
            State("resample-rate", "value"),
            State('model-output-files', 'children'),
            State('EEG-graph', 'figure'), State('bad-channels-dropdown', 'value')
        ]
    )
    def _update_EEG_plot(plot_button, redraw_button, point_clicked,
                            scale, channel_offset, segment_size, use_slider,
                            annotation_label, annotation_label_color, 
                            reset_models, run_model_bool, model_annotate, model_threshold, show_annotations_only,
                            current_file_name, selected_channels,
                            high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation,
                            resample_rate,
                            model_output_files,
                            current_fig, current_selected_bad_channels):
        """Generates EEG plot preprocessed with given parameter values. Triggered when plot-, redraw-, left-arrow-, and right-arrow button are clicked.

        Args:
            plot_button (int): Num clicks on plot button.
            redraw_button (int): Num clicks on redraw button.
            point_clicked (dict): Data from latest click event.
            scale (float): Input desired scaling for data.
            channel_offset (float): Input desired channel offset.
            segment_size (int): Input desired segment size for plots.
            use_slider (bool): Whether or not to activate view-slider.
            annotation_label (string); Label for new annotations.
            annotation_label_color (dict); Color for new annotations.
            reset_models (int): Num clicks on reset-models buttons.
            run_model_bool (list): List containing 1 if running integrated model is chosen.
            model_annotate (list): List containing 1 if automatic annotation is chosen.
            model_threshold (float): Input desired confidence threshold over which to automatically annotate.
            show_annotations_only (bool): Whether or not to only show annotations.
            current_file_name (string): File-name of loaded EEG recording.
            selected_channels (list): List of strings of channels selected for plotting.
            high_pass (float): Input desired high-pass filter value.
            low_pass (float): Input desired low-pass filter value.
            reference (string): Chosen reference.
            bad_channel_detection (string): Chosen automatic bad-channel detection.
            bad_channel_interpolation (list): List containing 1 if bad-channel interpolation is chosen.
            resample_rate (int): Input desired sampling frequency.
            model_output_files (list): List of strings of model-output file-names.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.
            current_selected_bad_channels (list): List containing names of currently selected bad channels.

        Returns:
            plotly.graph_objs.Figure: EEG plot.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        globals.preloaded_plots = {}
        
        if 'annotation-label' in trigger or 'annotation-label-color' in trigger:
            if globals.plotting_data:
                print(annotation_label_color)
                current_fig['layout']['newshape']['fillcolor'] = annotation_label_color
                
                current_fig['layout']['shapes'] = []
                for annotation in globals.marked_annotations:
                    current_fig['layout']['shapes'].append({
                        'editable': True,
                        'xref': 'x',
                        'yref': 'y',
                        'layer': 'below',
                        'opacity': 0.6,
                        'line': {'width': 0},
                        'fillcolor': globals.annotation_label_colors[annotation[2]],
                        'fillrule': 'evenodd',
                        'type': 'rect',
                        'x0': annotation[0],
                        'y0': len(globals.plotting_data['EEG']['channel_names']) * globals.plotting_data['plot']['offset_factor'] + globals.plotting_data['plot']['offset_factor'],
                        'x1': annotation[1],
                        'y1': -1 * len(globals.plotting_data['model']) * globals.plotting_data['plot']['offset_factor'] - globals.plotting_data['plot']['offset_factor']
                    })

                return current_fig

        if 'scale' in trigger or 'channel-offset' in trigger:
            if globals.plotting_data:
                if 'scale' in trigger and globals.plotting_data['EEG']['scaling_factor'] != scale:
                    new_scale = _get_scaling(scale)
                    print(new_scale)
                    
                    globals.plotting_data['EEG']['scaling_factor'] = new_scale
                    
                if 'channel-offset' in trigger and globals.plotting_data['plot']['offset_factor'] != channel_offset:
                    new_offset = _get_offset(channel_offset)
                    print(new_offset)
                    
                    globals.plotting_data['plot']['offset_factor'] = new_offset
                    
                    for model_index, model_array in enumerate(globals.plotting_data['model']):
                        globals.plotting_data['model'][model_index]['offset_model_data'] = [-((2 + model_index) * (globals.plotting_data['plot']['offset_factor'])) for i in range(len(globals.plotting_data['model'][model_index]['model_timescale']))]

                    y_ticks_model_output = np.arange((-len(globals.plotting_data['model']) - 1), -1)
                    y_ticks_channels = np.arange(0, len(globals.plotting_data['EEG']['channel_names']))
                    y_ticks = np.concatenate((y_ticks_model_output, y_ticks_channels))
                    y_ticks = y_ticks * (globals.plotting_data['plot']['offset_factor'])

                    globals.plotting_data['plot']['y_ticks'] = y_ticks

                offset_EEG = globals.plotting_data['EEG']['EEG_data'].copy() 
                offset_EEG = offset_EEG * globals.plotting_data['EEG']['scaling_factor']
                
                for channel_index in range(offset_EEG.shape[1]):
                    # Calculate offset for y-axis
                    offset_EEG[:, channel_index] = offset_EEG[:, channel_index] + ((globals.plotting_data['plot']['offset_factor']) * (len(globals.plotting_data['EEG']['channel_names']) - 1 - channel_index))  # First channel goes to top of the plot

                globals.plotting_data['EEG']['offset_EEG_data'] = offset_EEG

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig

        if 'segment-size' in trigger:
            if globals.plotting_data:
                if segment_size:
                    globals.x1 = globals.x0 + segment_size + 1
                else:
                    globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig

        if 'use-slider' in trigger:
            if globals.plotting_data:
                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig

        if 'clickData' in trigger:
            channel_index = point_clicked['points'][0]['curveNumber']
            if channel_index >= len(globals.plotting_data['EEG']['channel_names']):
                return current_fig

            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]

            if channel_name not in current_selected_bad_channels:
                current_selected_bad_channels.append(channel_name)
            else:
                current_selected_bad_channels.remove(channel_name)

            globals.raw.info['bads'] = current_selected_bad_channels
            print(current_selected_bad_channels)

            for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
                channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                
                if channel_name in current_selected_bad_channels:
                    globals.plotting_data['EEG']['default_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
                    globals.plotting_data['EEG']['channel_visibility'][channel_index] = False
                    globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
                else:
                    globals.plotting_data['EEG']['default_channel_colors'][channel_index] = 'black'
                    globals.plotting_data['EEG']['channel_visibility'][channel_index] = True
                    globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'black'

                current_fig['data'][channel_index]['marker']['color'] = globals.plotting_data['EEG']['default_channel_colors'][channel_index]

            current_fig['layout']['updatemenus'][0]['buttons'][2]['args'][0]['visible'] = globals.plotting_data['EEG']['channel_visibility']
            current_fig['layout']['updatemenus'][0]['buttons'][2]['args2'][0]['visible'] = True

            if len(current_fig['layout']['updatemenus'][0]['buttons']) > 3:
                for model_index in range(len(globals.plotting_data['model'])):
                    if globals.plotting_data['model'][model_index]['model_channels']:
                        for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
                            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                            if channel_name in globals.plotting_data['model'][model_index]['model_channels']:
                                globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'blue'
                                
                            current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'][channel_index] = globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index]
                            current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'][channel_index] = globals.plotting_data['EEG']['default_channel_colors'][channel_index]

            return current_fig

        # If re-drawing, keep current annotations and bad channels
        if ('redraw-button' in trigger or 'run-model' in trigger) and run_model_bool:
            if globals.plotting_data:
                globals.model_raw.info['bads'] = current_selected_bad_channels

                print('Running model...')
                run_model_output, run_model_channel_names, run_model_sample_rate, run_model_description = run_model(globals.model_raw.copy(), globals.viewing_raw.copy())

                # Model annotations
                if model_annotate:
                    all_model_annotations = []

                    if run_model_sample_rate:
                        model_timestep = 1 / run_model_sample_rate
                    else:
                        model_timestep = 1 / globals.model_raw.info['sfreq']
                    # print(model_timestep)

                    if not model_threshold:
                        model_threshold = 0.7

                    model_annotations = confidence_intervals(model, model_threshold, 1, model_timestep)
                    for interval_index, interval in enumerate(model_annotations):
                        model_annotations[interval_index] = (interval[0], interval[1], run_model_description)

                    all_annotations = globals.marked_annotations + model_annotations
                    all_annotations = merge_intervals(all_annotations)

                    globals.marked_annotations = all_annotations

                    annotations_to_raw(globals.raw, globals.marked_annotations)
                    annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
                
                if not globals.plotting_data['model']:
                    globals.plotting_data['model'].append({})
                    globals.plotting_data['EEG']['default_channel_colors'].append(None)
                    globals.plotting_data['EEG']['highlighted_channel_colors'].append(None)
                    globals.plotting_data['EEG']['channel_visibility'].append(True)

                globals.plotting_data['model'][-1]['model_data'] = run_model_output
                globals.plotting_data['model'][-1]['model_channels'] = run_model_channel_names
                globals.plotting_data['model'][-1]['model_timescale'] = np.linspace(0, globals.plotting_data['EEG']['recording_length'], num=run_model_output.shape[0])
                globals.plotting_data['model'][-1]['offset_model_data'] = [-((2 + len(globals.plotting_data['model']) - 1) * (globals.plotting_data['plot']['offset_factor'])) for i in range(len(globals.plotting_data['model'][-1]['model_timescale']))]

                globals.plotting_data['EEG']['default_channel_colors'][-1] = run_model_output
                globals.plotting_data['EEG']['highlighted_channel_colors'][-1] = run_model_output
                
                # current_fig['data'][-1]['marker']['color'] = run_model_output
                
                # current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'] = globals.plotting_data['EEG']['highlighted_channel_colors']
                # current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'] = globals.plotting_data['EEG']['default_channel_colors']

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig
            else:
                return current_fig

        if 'reset-models' in trigger:
            if globals.plotting_data and globals.plotting_data['model']:
                del globals.plotting_data['model'][:-1]

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig
            else:
                return current_fig

        if 'run-model' in trigger and not run_model_bool:
            if globals.plotting_data and globals.plotting_data['model']:
                del globals.plotting_data['model'][-1]

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

                return updated_fig
            else:
                return current_fig

        if 'model-threshold' in trigger or 'annotate-model' in trigger:
            if globals.plotting_data:
                all_model_annotations = []
                
                if model_annotate:
                    for model in globals.plotting_data['model']:
                        model_timestep = model['model_timescale'][1]

                        output_intervals = confidence_intervals(model['model_data'], model_threshold, 1, model_timestep)
                        for interval_index, interval in enumerate(output_intervals):
                            output_intervals[interval_index] = (interval[0], interval[1], 'bad_artifact_model')
                        all_model_annotations = all_model_annotations + output_intervals

                remaining_annotations = [annotation for annotation in globals.marked_annotations if annotation[2] != 'bad_artifact_model']

                merged_annotations = merge_intervals(all_model_annotations + remaining_annotations)

                globals.marked_annotations = merged_annotations

                annotations_to_raw(globals.raw, globals.marked_annotations)
                annotations_to_raw(globals.viewing_raw, globals.marked_annotations)

                if show_annotations_only and len(globals.marked_annotations) > 0:
                    globals.current_plot_index = 0

                    globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                    globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2

                    updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)
                    return updated_fig
                else:
                    current_fig['layout']['shapes'] = []
                    for annotation in globals.marked_annotations:
                        current_fig['layout']['shapes'].append({
                            'editable': True,
                            'xref': 'x',
                            'yref': 'y',
                            'layer': 'below',
                            'opacity': 0.6,
                            'line': {'width': 0},
                            'fillcolor': globals.annotation_label_colors[annotation[2]],
                            'fillrule': 'evenodd',
                            'type': 'rect',
                            'x0': annotation[0],
                            'y0': len(globals.plotting_data['EEG']['channel_names']) * globals.plotting_data['plot']['offset_factor'] + globals.plotting_data['plot']['offset_factor'],
                            'x1': annotation[1],
                            'y1': -1 * len(globals.plotting_data['model']) * globals.plotting_data['plot']['offset_factor'] - globals.plotting_data['plot']['offset_factor']
                        })

                    return current_fig
        
        # if 'annotate-model' in trigger and not model_annotate:
        #     if globals.plotting_data:
        #         if run_model_bool or model_output_files:
        #             print(current_fig['data'])
                
        
        if 'show-annotations-only' in trigger:
            if globals.plotting_data:
                globals.current_plot_index = 0

                if show_annotations_only:
                    if globals.marked_annotations:
                        globals.x0 = globals.marked_annotations[0][0] - 2
                        globals.x1 = globals.marked_annotations[0][1] + 2
                    else:
                        print('No annotations found')
                        show_annotations_only = False
                else:
                    globals.x0 = -0.5
                    if segment_size:
                        globals.x1 = segment_size + 0.5
                    else:
                        globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)
                
                return updated_fig

        if 'plot-button' in trigger:
            globals.current_plot_index = 0
            
            globals.x0 = -0.5
            if segment_size:
                globals.x1 = segment_size + 0.5
            else:
                globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

            print('Loading data...')

            if globals.external_raw:
                globals.raw = globals.external_raw.copy()
            elif not globals.external_raw:
                globals.raw = parse_data_file(current_file_name)  # reload data in case preprocessing has changed

            globals.marked_annotations = get_annotations(globals.raw)
            
            globals.raw.info['bads'] = current_selected_bad_channels

            # if run_model_bool:
            globals.model_raw = globals.raw.copy()

            # MNE preprocessing
            print('Pre-processing data...')

            globals.raw = preprocess_EEG(globals.raw, high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation)
            
            if run_model_bool:
                globals.model_raw.info['bads'] = globals.raw.info['bads']

            # Resampling
            globals.viewing_raw = globals.raw.copy()

            if resample_rate and float(resample_rate) < globals.raw.info['sfreq']:
                print('Resample-rate: {}'.format(resample_rate))
                print('Performing resampling')
                globals.viewing_raw.resample(resample_rate)
                # timestep = 1 / resample_rate

            print(globals.viewing_raw.info)

            if selected_channels:
                selected_channel_names = selected_channels
                print(selected_channel_names)
            else:
                selected_channel_names = []
                print('No specific channels selected')

            model_output = []
            model_channel_names = []
            model_sample_rate = []
            model_descriptions = []
            if model_output_files:
                for model_name in model_output_files:
                    if '.csv' in model_name:
                        loaded_annotations = parse_annotation_file(model_name)
                        merged_annotations = merge_intervals(globals.marked_annotations + loaded_annotations)

                        globals.marked_annotations = merged_annotations
                        annotations_to_raw(globals.raw, globals.marked_annotations)
                        annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
                    else:
                        temp_model_output, temp_channel_names, temp_sample_rate, temp_descriptions = parse_model_output_file(model_name, globals.raw)
                        model_output.append(temp_model_output)
                        model_channel_names.append(temp_channel_names)
                        model_sample_rate.append(temp_sample_rate)
                        model_descriptions.append(temp_descriptions)

            if run_model_bool:
                print('Running model...')
                run_model_output, run_model_channel_names, run_model_sample_rate, run_model_description = run_model(globals.model_raw.copy(), globals.viewing_raw.copy())
                model_output.append(run_model_output)
                model_channel_names.append(run_model_channel_names)
                model_sample_rate.append(run_model_sample_rate)
                model_descriptions.append(run_model_description)

            if (not (model_output_files or run_model_bool)) and model_annotate:
                print('No model selected to annotate with!')
                model_annotate = False

            # Model annotations
            if model_annotate:
                all_model_annotations = []
                for model_index, model in enumerate(model_output):
                    if model_sample_rate[model_index]:
                        model_timestep = 1 / model_sample_rate[model_index]
                    else:
                        model_timestep = 1 / globals.model_raw.info['sfreq']
                    # print(model_timestep)
                    if not model_threshold:
                        model_threshold = 0.7
                    output_intervals = confidence_intervals(model, model_threshold, 1, model_timestep)
                    for interval_index, interval in enumerate(output_intervals):
                        output_intervals[interval_index] = (interval[0], interval[1], model_descriptions[model_index])
                    all_model_annotations = all_model_annotations + output_intervals

                merged_model_annotations = merge_intervals(all_model_annotations)

                all_annotations = globals.marked_annotations + merged_model_annotations
                all_annotations = merge_intervals(all_annotations)

                globals.marked_annotations = all_annotations

                annotations_to_raw(globals.raw, globals.marked_annotations)
                annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
            
            if show_annotations_only:
                if globals.marked_annotations:
                    globals.x0 = globals.marked_annotations[0][0] - 2
                    globals.x1 = globals.marked_annotations[0][1] + 2
                else:
                    print('No annotations found')
                    show_annotations_only = False

            fig = get_EEG_figure(current_file_name, globals.viewing_raw, selected_channel_names, annotation_label, scale, channel_offset, model_output, model_channel_names, use_slider, show_annotations_only)
            
            return fig

        # Default plot when app is opened
        else:
            # fig = Figure()  # empty figure
            img = io.imread(c.TITLE_IMAGE_FILE)
            fig = px.imshow(img)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_traces(hovertemplate=None, hoverinfo='skip')
            return fig

    # Data selection returning power-spectrum callback
    @app.callback(
        [Output('selected-data', 'children'), Output('power-spectrum', 'figure')],
        [Input('EEG-graph', 'selectedData')]
    )
    def _get_selected_power_spectrum(selectedData):
        """Calculates frequency with highest power density and power-spectrum plot of selectedData.

        Args:
            selectedData (dict): Data from latest selection event.

        Returns:
            tuple(string, plotly.graph_objs.Figure): String of frequency with highest power density, power-spectrum plot of selectedData.
        """
        if not selectedData or (not selectedData['points']):
            most_prominent_freq = '-'
            fig = Figure()
        else:
            # print(selectedData)
            # selected_data = []

            trace_number = selectedData['points'][0]['curveNumber']
            # print('First trace: {}'.format(trace_number))

            selected_range = selectedData['range']
            print('Range: {}'.format(selected_range))

            split_dict = collections.defaultdict(list)

            for datapoint in selectedData['points']:
                split_dict[datapoint['curveNumber']].append(datapoint['customdata'])

            selected_traces_list = list(split_dict.values())

            sample_rate = globals.viewing_raw.info['sfreq']

            all_Pxx_den = []

            for counter, trace in enumerate(selected_traces_list):
                # print(counter)
                f, Pxx_den = calc_power_spectrum(sample_rate, trace)
                all_Pxx_den.append(Pxx_den)

            mean_Pxx_den = np.mean(all_Pxx_den, axis=0)

            most_prominent_freq = get_most_prominent_freq(f, mean_Pxx_den)
            most_prominent_freq = round(most_prominent_freq, 2)

            fig = get_power_spectrum_plot(f, mean_Pxx_den)

        return (str(most_prominent_freq) + ' Hz'), fig
