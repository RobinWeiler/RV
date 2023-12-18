import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px

from skimage import io

import numpy as np

from helperfunctions.annotation_helperfunctions import merge_intervals, get_annotations, annotations_to_raw, confidence_intervals
from helperfunctions.loading_helperfunctions import parse_data_file, parse_model_output_file, parse_annotation_file
from helperfunctions.preprocessing_helperfunctions import preprocess_EEG
from helperfunctions.visualization_helperfunctions import get_EEG_figure, get_EEG_plot, _get_scaling, _get_offset
from model.run_model import run_model

import constants as c
import globals


def register_visualization_callbacks(app):

    # plot callback
    @app.callback(
        [Output('EEG-graph', 'figure'), Output('EEG-graph', 'style'),],
        [
            Input('plot-button', 'n_clicks'),
            Input("confirm-plot-button", "n_clicks"),
            Input("resample-rate", "value"),
            Input("scale", "value"),
            Input("channel-offset", "value"),
            Input('use-slider', 'value'),
            Input('skip-hoverinfo', 'value'),
            Input('show-annotations-only', 'value'),
        ],
        [
            State('data-file', 'children'),
            State('selected-channels-dropdown', 'value'),
            State("high-pass", "value"), State("low-pass", "value"),
            State('reference-dropdown', 'value'),
            State('bad-channel-detection-dropdown', 'value'), State("bad-channel-interpolation", "value"),
            State('segment-size', 'value'),
            State('annotation-label', 'value'),
            State('show-annotation-labels', 'value'),
            State('upload-model-output', 'filename'),
            State("run-model", "value"),
            State("annotate-model", "value"),
            State("model-threshold", "value"),
            State('hide-bad-channels-button', 'n_clicks'),
            State('highlight-model-channels-button', 'n_clicks'),
            State('EEG-graph', 'figure'), State('bad-channels-dropdown', 'value'),
            State("hidden-bandpass-changed", "n_clicks"),
        ]
    )
    def _update_EEG_plot(
        plot_button, confirm_plot_button,
        resample_rate, scale, channel_offset, use_slider, skip_hoverinfo,
        show_annotations_only,
        current_file_name, selected_channels,
        high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation,
        segment_size,
        annotation_label, show_annotation_labels,
        model_output_files, run_model_bool, model_annotate, model_threshold,
        hide_bad_channels, highlight_model_channels,
        current_fig, current_selected_bad_channels,
        bandpass_changed
    ):
        """Generates EEG plot preprocessed with given parameter values. Triggered when plot-, redraw-, left-arrow-, and right-arrow button are clicked.

        Args:
            plot_button (int): Num clicks on plot button.
            resample_rate (int): Input desired sampling frequency.
            scale (float): Input desired scaling for data.
            channel_offset (float): Input desired channel offset.
            use_slider (bool): Whether or not to activate view-slider.
            skip_hoverinfo (bool): Whether or not to activate hover-info.
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
            segment_size (int): Input desired segment size for plots.
            annotation_label (string); Label for new annotations.
            model_output_files (list): List of strings of model-output file-names.
            hide_bad_channels (int): Num clicks on hide-bad-channels-button button.
            highlight_model_channels (int): Num clicks on highlight-model-channels-button button.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.
            current_selected_bad_channels (list): List containing names of currently selected bad channels.

        Returns:
            plotly.graph_objs.Figure: EEG plot.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print('visualization trigger: {}'.format(trigger))

        globals.preloaded_plots = {}

        if not use_slider:
            fig_style = {'height': '97vh'}
        else:
            fig_style = {'height': '90vh'}

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

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

                return updated_fig, fig_style

        if 'resample-rate' in trigger:
            if globals.raw and globals.plotting_data:
                globals.viewing_raw = globals.raw.copy()

                if resample_rate and float(resample_rate) < globals.raw.info['sfreq']:
                    print('Resample-rate: {}'.format(resample_rate))
                    print('Performing resampling')
                    globals.viewing_raw.resample(resample_rate)
                    # timestep = 1 / resample_rate

                print(globals.viewing_raw.info)

                globals.plotting_data['EEG']['recording_length'] = len(globals.viewing_raw) / globals.viewing_raw.info['sfreq']

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

                return updated_fig, fig_style

        if 'use-slider' in trigger:
            if globals.plotting_data:
                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

                return updated_fig, fig_style

        if 'skip-hoverinfo' in trigger:
            if globals.plotting_data:
                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

                return updated_fig, fig_style

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

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))
                
                return updated_fig, fig_style

        if 'plot-button' in trigger:
            if 'plot-button.n_clicks' == trigger and bandpass_changed or plot_button == 0:
                raise PreventUpdate

            globals.current_plot_index = 0
            
            globals.x0 = -0.5
            if segment_size:
                globals.x1 = segment_size + 0.5
            else:
                globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

            print('Loading data...')

            if 'plot-button.n_clicks' == trigger and globals.raw:
                globals.marked_annotations = get_annotations(globals.raw)

            if not globals.external_raw:
                globals.raw = parse_data_file(current_file_name)  # reload data in case preprocessing has changed

            if not ('plot-button.n_clicks' == trigger and globals.raw):
                globals.marked_annotations = get_annotations(globals.raw)
            else:
                globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)
            
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

            if 'annotations' in globals.parameters.keys():
                loaded_annotations = globals.parameters['annotations']
                merged_annotations, _ = merge_intervals(globals.marked_annotations + loaded_annotations)
                globals.marked_annotations = merged_annotations
                annotations_to_raw(globals.raw, globals.marked_annotations)
                annotations_to_raw(globals.viewing_raw, globals.marked_annotations)

            model_output = []
            model_channel_names = []
            model_sample_rate = []
            model_descriptions = []
            if model_output_files:
                for model_name in model_output_files:
                    if '.csv' in model_name:
                        loaded_annotations = parse_annotation_file(model_name)
                        merged_annotations, _ = merge_intervals(globals.marked_annotations + loaded_annotations)
                        globals.marked_annotations = merged_annotations
                        annotations_to_raw(globals.raw, globals.marked_annotations)
                        annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
                    else:
                        temp_model_output, temp_channel_names, temp_sample_rate, temp_descriptions = parse_model_output_file(model_name, globals.viewing_raw)
                        model_output.append(temp_model_output)
                        model_channel_names.append(temp_channel_names)
                        model_sample_rate.append(temp_sample_rate)
                        model_descriptions.append(temp_descriptions)

            if run_model_bool:
                print('Running model...')
                run_model_output, run_model_channel_names, run_model_sample_rate, run_model_description = run_model(globals.model_raw, globals.viewing_raw)
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

                merged_model_annotations, _ = merge_intervals(all_model_annotations)

                all_annotations = globals.marked_annotations + merged_model_annotations
                all_annotations, _ = merge_intervals(all_annotations)

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

            fig = get_EEG_figure(current_file_name, globals.viewing_raw, selected_channels, annotation_label, show_annotation_labels, scale, channel_offset, model_output, model_channel_names, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))
            
            return fig, fig_style

        # Default plot when app is opened
        # fig = Figure()  # empty figure
        img = io.imread(c.TITLE_IMAGE_FILE)
        fig = px.imshow(img)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hovertemplate=None, hoverinfo='skip')
        return fig, fig_style
