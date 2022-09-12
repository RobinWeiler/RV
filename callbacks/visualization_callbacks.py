import json
import math
import collections
import numpy as np

import dash
from dash.dependencies import Input, Output, State

from plotly.graph_objs import Figure
import plotly.express as px
from skimage import io

from helperfunctions.annotation_helperfunctions import merge_intervals, get_annotations, annotations_to_raw, confidence_intervals
from helperfunctions.loading_helperfunctions import parse_data_file, parse_model_output_file
from helperfunctions.visualization_helperfunctions import get_EEG_figure, calc_power_spectrum, get_most_prominent_freq, get_power_spectrum_plot, get_EEG_plot, preprocess_EEG
from model.run_model import run_model

import constants as c
import globals


def register_visualization_callbacks(app):
    # Selecting channels to plot callback
    @app.callback(
        [Output('selected-channels-dropdown', 'value'), Output('channel-topography', 'selectedData')],
        [Input('channel-topography', 'selectedData'), Input('data-file', 'children')],
        # Input('selected-channels-dropdown', 'value'),
        # prevent_initial_call=True
    )
    def _get_selected_channels(selectedData, file_name):
        """Retrieves names of selected channels. Triggered when datapoints are selected in channel-topography plot and when new file is loaded.

        Args:
            selectedData (dict): Data from latest selection event.
            file_name (string): File-name.

        Returns:
            tuple(list, dict): List of strings of channels selected for plotting, empty dict to reset selectedData.
        """
        # print(json.dumps(selectedData, indent=2))
        selected_channels = []

        if selectedData:
            for selected_channel in selectedData['points']:
                # print(selected_channel['customdata'])
                selected_channels.append(selected_channel['customdata'])

        return selected_channels, {}

    @app.callback(
        [Output('left-button', 'disabled'), Output('right-button', 'disabled')],
        Input('EEG-graph', 'figure'),
        State('segment-size', 'value'), 
        # prevent_initial_call=True
    )
    def _update_arrow_buttons(fig, segment_size):
        """Disables/enables arrow-buttons based on position of current segment. Triggered when EEG plot has loaded.

        Args:
            segment_size (int): Segment size of EEG plot.
            fig (plotly.graph_objs.Figure): EEG plot.

        Returns:
            tuple(bool, bool): Whether or not to disable left-arrow button, whether or not to disable right-arrow button.
        """
        if segment_size and globals.plotting_data:
            if globals.x0 == -0.5 and not globals.x1 > globals.plotting_data['EEG']['recording_length']:
                return True, False
            elif globals.x1 > globals.plotting_data['EEG']['recording_length']:
                return False, True
            else:
                return False, False
        else:
            return True, True

    @app.callback(
        Output('preload-data', 'children'),
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('use-slider', 'value')],
        prevent_initial_call=True
    )
    def _preload_plots(fig, segment_size, use_slider):
        """Preloads 1 following segment and adds it to globals.preloaded_plots. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            use_slider (bool): Whether or not to activate view-slider.
        """
        if globals.plotting_data:
            if segment_size:
                print('Preloading segments...')
                num_segments = math.ceil(globals.plotting_data['EEG']['recording_length'] / segment_size)
                # print(num_segments)
                
                upper_bound = globals.current_plot_index + 2 if globals.current_plot_index + 2 < num_segments else num_segments
                # print(upper_bound)

                globals.preloaded_plots[globals.current_plot_index] = fig

                for segment_index in range(upper_bound):
                    if segment_index not in globals.preloaded_plots:
                        new_x0 = segment_index * segment_size - 0.5
                        new_x1 = segment_size + segment_index * segment_size + 0.5
                        globals.preloaded_plots[segment_index] = get_EEG_plot(globals.plotting_data, new_x0, new_x1, use_slider)
                        # print(segment_index)

    # plot-, redraw-, left-arrow-, and right-arrow-button callback
    @app.callback(
        Output('EEG-graph', 'figure'),
        [Input('plot-button', 'n_clicks'), Input('redraw-button', 'n_clicks'), Input('left-button', 'n_clicks'), Input('right-button', 'n_clicks'), Input('EEG-graph', 'clickData')],
        [
            State('data-file', 'children'),
            State('selected-channels-dropdown', 'value'),
            State("high-pass", "value"), State("low-pass", "value"),
            State('reference-dropdown', 'value'),
            State('bad-channel-detection-dropdown', 'value'), State("bad-channel-interpolation", "value"),
            State("resample-rate", "value"), State("scale", "value"), State("channel-offset", "value"), State('segment-size', 'value'), State('use-slider', 'value'),
            State('model-output-files', 'children'), State("run-model", "value"), State("annotate-model", "value"), State("model-threshold", "value"),
            State('EEG-graph', 'figure')
        ]
    )
    def _update_EEG_plot(plot_button, redraw_button, left_button, right_button, point_clicked, current_file_name, selected_channels,
                            high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation,
                            resample_rate, scale, channel_offset, segment_size, use_slider,
                            model_output_files, model_run, model_annotate, model_threshold, 
                            current_fig):
        """Generates EEG plot preprocessed with given parameter values. Triggered when plot-, redraw-, left-arrow-, and right-arrow button are clicked.

        Args:
            plot_button (int): Num clicks on plot button.
            redraw_button (int): Num clicks on redraw button.
            left_button (int): Num clicks on left-arrow button.
            right_button (int): Num clicks on right-arrow button.
            point_clicked (dict): Data from latest click event.
            current_file_name (string): File-name of loaded EEG recording.
            selected_channels (list): List of strings of channels selected for plotting.
            high_pass (float): Input desired high-pass filter value.
            low_pass (float): Input desired low-pass filter value.
            reference (string): Chosen reference.
            bad_channel_detection (string): Chosen automatic bad-channel detection.
            bad_channel_interpolation (list): List containing 1 if bad-channel interpolation is chosen.
            resample_rate (int): Input desired sampling frequency.
            scale (float): Input desired scaling for data.
            channel_offset (float): Input desired channel offset.
            segment_size (int): Input desired segment size for plots.
            use_slider (bool): Whether or not to activate view-slider.
            model_output_files (list): List of strings of model-output file-names.
            model_run (list): List containing 1 if running integrated model is chosen.
            model_annotate (list): List containing 1 if automatic annotation is chosen.
            model_threshold (float): Input desired confidence threshold over which to automatically annotate.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            plotly.graph_objs.Figure: EEG plot.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)

        if 'right-button' in trigger:
            if segment_size:
                globals.current_plot_index += 1

                globals.x0 += segment_size
                globals.x1 += segment_size
                
                # print(globals.x0, globals.x1)

                if globals.current_plot_index in globals.preloaded_plots:
                    updated_fig = globals.preloaded_plots[globals.current_plot_index]
                else:
                    updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, use_slider=use_slider)

                return updated_fig
        
        elif 'left-button' in trigger:
            if segment_size:
                globals.current_plot_index -= 1
                
                globals.x0 -= segment_size
                globals.x1 -= segment_size
                
                # print(globals.x0, globals.x1)

                if globals.current_plot_index in globals.preloaded_plots:
                    updated_fig = globals.preloaded_plots[globals.current_plot_index]
                else:
                    updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, use_slider=use_slider)

                return updated_fig

        globals.preloaded_plots = {}

        # If re-drawing, keep current annotations and bad channels
        if 'clickData' in trigger:
            channel_index = point_clicked['points'][0]['curveNumber']
            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
            
            current_selected_bad_channels = globals.raw.info['bads']

            if channel_name not in current_selected_bad_channels:
                current_selected_bad_channels.append(channel_name)
            else:
                current_selected_bad_channels.remove(channel_name)

            globals.raw.info['bads'] = current_selected_bad_channels
            print(current_selected_bad_channels)

            for channel_index in range(len(globals.plotting_data['EEG']['channel_names']) - len(globals.plotting_data['model'])):
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
                        for channel_index in range(len(globals.plotting_data['EEG']['channel_names']) - len(globals.plotting_data['model'])):
                            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                            if channel_name in globals.plotting_data['model'][model_index]['model_channels']:
                                globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'blue'
                                
                            current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'][channel_index] = globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index]
                            current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'][channel_index] = globals.plotting_data['EEG']['default_channel_colors'][channel_index]

            return current_fig

        if 'redraw-button' in trigger:
            # globals.model_raw.info['bads'] = current_selected_bad_channels

            # print('Running model...')
            # run_model_output, run_model_channel_names, run_model_sample_rate = run_model(globals.model_raw.copy(), globals.viewing_raw.copy())

            # # Model annotations
            # if model_annotate:
            #     all_model_annotations = []

            #     if run_model_sample_rate:
            #         model_timestep = 1 / run_model_sample_rate
            #     else:
            #         model_timestep = 1 / globals.model_raw.info['sfreq']
            #     # print(model_timestep)

            #     if not model_threshold:
            #         model_threshold = 0.7

            #     model_annotations = confidence_intervals(model, model_threshold, 1, model_timestep)

            #     all_annotations = globals.marked_annotations + model_annotations
            #     all_annotations = merge_intervals(all_annotations)

            #     globals.marked_annotations = all_annotations

            #     annotations_to_raw(globals.raw, globals.marked_annotations)
            #     annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
            
            # globals.plotting_data['model'][-1]['model_data'] = run_model_output
            # globals.plotting_data['model'][-1]['model_channels'] = run_model_channel_names
            # globals.plotting_data['model'][-1]['model_timescale'] = np.linspace(0, globals.plotting_data['EEG']['recording_length'], num=run_model_output.shape[0])
            # globals.plotting_data['model'][-1]['offset_model_data'] = [-((2 + len(globals.plotting_data['model'])) * (globals.plotting_data['plot']['offset_factor'])) for i in range(len(globals.plotting_data['model'][-1]['model_timescale']))]

            # globals.plotting_data['EEG']['default_channel_colors'][-1] = run_model_output
            # globals.plotting_data['EEG']['highlighted_channel_colors'][-1] = run_model_output
            
            # current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'] = globals.plotting_data['EEG']['highlighted_channel_colors']
            # current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'] = globals.plotting_data['EEG']['default_channel_colors']

            return current_fig

        elif 'plot-button' in trigger:
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

            if model_run:
                globals.model_raw = globals.raw.copy()

            # MNE preprocessing
            print('Pre-processing data...')

            globals.raw = preprocess_EEG(globals.raw, high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation)
            
            if model_run:
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
            if model_output_files:
                for model_name in model_output_files:
                    # print(model_name)
                    temp_model_output, temp_channel_names, temp_sample_rate = parse_model_output_file(model_name)
                    model_output.append(temp_model_output)
                    model_channel_names.append(temp_channel_names)
                    model_sample_rate.append(temp_sample_rate)

            if model_run:
                print('Running model...')
                run_model_output, run_model_channel_names, run_model_sample_rate = run_model(globals.model_raw.copy(), globals.viewing_raw.copy())
                model_output.append(run_model_output)
                model_channel_names.append(run_model_channel_names)
                model_sample_rate.append(run_model_sample_rate)

            if (not (model_output_files or model_run)) and model_annotate:
                print('No model selected to annotate with!')
                model_annotate = False

            # Model annotations
            if model_annotate:
                all_model_annotations = []
                for i, model in enumerate(model_output):
                    if model_sample_rate[i]:
                        model_timestep = 1 / model_sample_rate[i]
                    else:
                        model_timestep = 1 / globals.model_raw.info['sfreq']
                    # print(model_timestep)
                    if not model_threshold:
                        model_threshold = 0.7
                    output_intervals = confidence_intervals(model, model_threshold, 1, model_timestep)
                    all_model_annotations = all_model_annotations + output_intervals

                merged_model_annotations = merge_intervals(all_model_annotations)

                all_annotations = globals.marked_annotations + merged_model_annotations
                all_annotations = merge_intervals(all_annotations)

                globals.marked_annotations = all_annotations

                annotations_to_raw(globals.raw, globals.marked_annotations)
                annotations_to_raw(globals.viewing_raw, globals.marked_annotations)

            fig = get_EEG_figure(current_file_name, globals.viewing_raw, selected_channel_names, EEG_scale=scale, channel_offset=channel_offset, model_output=model_output, model_channels=model_channel_names, use_slider=use_slider)
            
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
