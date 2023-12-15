import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

import numpy as np

from helperfunctions.annotation_helperfunctions import confidence_intervals, merge_intervals, annotations_to_raw
from helperfunctions.visualization_helperfunctions import get_EEG_plot, _get_list_for_displaying
from model.run_model import run_model

import globals
import constants as c


def register_model_callbacks(app):

    # Select model output
    @app.callback(
        [Output('model-output-files', 'children'), Output('upload-model-output', 'filename')],
        [Input('upload-model-output', 'filename'), Input('reset-models', 'n_clicks')],
        prevent_initial_call=True
    )
    def _update_model_output_files(list_selected_file_names, reset_models):
        """Retrieves file-names of selected model-output files. Triggers when new files are loaded or reset-models button is clicked. The latter removes selected files.

        Args:
            list_selected_file_names (list): List of strings of selected model-output file-names.
            reset_models (int): Num clicks on reset-models button.

        Returns:
            tuple(list, list): Both lists contain strings of selected model-output file-names.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'reset-models' in trigger:
            print('Resetting loaded model-predictions')
            return [], None
        elif list_selected_file_names:
            print('Selected files: {}'.format(list_selected_file_names))
            return _get_list_for_displaying(list_selected_file_names), list_selected_file_names

    # Disable rerun-model button - comment this out for models that are not deterministic
    @app.callback(
        Output('rerun-model-button', 'disabled'),
        [Input('EEG-graph', 'clickData'), Input('rerun-model-button', 'n_clicks')], 
        [State("run-model", "value"), State('model-output-files', 'children')],
        # prevent_initial_call=True
    )
    def _disable_rerun_model_button(selected_bad_channel, rerun_model_button, run_model_bool, model_files):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'clickData' in trigger and globals.plotting_data and (run_model_bool or model_files):
            return False

        return True

    # Disable model threshold
    @app.callback(
        Output('model-threshold', 'disabled'),
        Input('annotate-model', 'value'),
        # prevent_initial_call=True
    )
    def _disable_model_threshold(model_annotate):
        return not model_annotate

    # Enable/disable Highlight model channels button
    @app.callback(
        Output('highlight-model-channels-button', 'disabled'),
        Input('EEG-graph', 'figure'),
        # prevent_initial_call=True
    )
    def _update_hide_bad_channels_button(fig):
        """Disables/enables higlight-model-channels-button. Triggered when selected bad channels change.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.

        Returns:
            bool: Whether or not to disable higlight-model-channels-button button.
        """
        if globals.plotting_data:
            return not any(globals.plotting_data['model'][model_index]['model_channels'] for model_index in range(len(globals.plotting_data['model'])))
        else:
            return True

    # Highlight model channels
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('highlight-model-channels-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def  _use_highlight_model_channels_button(highlight_model_channels):
        """Hides bad channels when pressed. Shows all channels when pressed again.

        Args:
            highlight_model_channels (int): Num clicks on hide-bad-channels-button button.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """

        if globals.plotting_data:
            patched_fig = Patch()
            for model_index in range(len(globals.plotting_data['model'])):
                for channel_name in globals.plotting_data['model'][model_index]['model_channels']:
                    channel_index = globals.plotting_data['EEG']['channel_names'].index(channel_name)

                    if highlight_model_channels % 2 != 0:
                        patched_fig['data'][channel_index]['marker']['color'] = c.MODEL_CHANNEL_COLOR
                    else:
                        patched_fig['data'][channel_index]['marker']['color'] = 'black'

                        if channel_name in globals.plotting_data['EEG']['eog_channels']:
                            patched_fig['data'][channel_index]['marker']['color'] = 'blue'
                        if channel_name in globals.raw.info['bads']:
                            patched_fig['data'][channel_index]['marker']['color'] = c.BAD_CHANNEL_COLOR

            return patched_fig
        else:
            raise PreventUpdate

    # Update plot when model settings are changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        [Input('rerun-model-button', 'n_clicks'), Input('reset-models', 'n_clicks'), Input("annotate-model", "value"), Input("model-threshold", "value")],
        [
            State("run-model", "value"),
            State('use-slider', 'value'), State('reorder-channels', 'value'), State('skip-hoverinfo', 'value'), 
            State('annotation-label', 'value'), State('show-annotation-labels', 'value'), State('show-annotations-only', 'value'), 
            State('hide-bad-channels-button', 'n_clicks'), State('highlight-model-channels-button', 'n_clicks'), State('bad-channels-dropdown', 'value'), 
            State('EEG-graph', 'figure')
        ],
        prevent_initial_call=True
    )
    def _update_EEG_plot_model(run_model_bool, rerun_model_button, reset_models_button, model_annotate, model_threshold, use_slider, reorder_channels, skip_hoverinfo, annotation_label, show_annotation_labels, show_annotations_only, hide_bad_channels, highlight_model_channels, current_selected_bad_channels, current_fig):
        """Updates plot when model settings are changed.

        Args:
            run_model_bool (list): List containing 1 if running integrated model is chosen.
            rerun_model_button (int): Num clicks on rerun-model button.
            reset_models_button (int): Num clicks on reset-models button.
            model_annotate (list): List containing 1 if automatic annotation is chosen.
            model_threshold (float): Input desired confidence threshold over which to automatically annotate.
            use_slider (bool): Whether or not to activate view-slider.
            skip_hoverinfo (bool): Whether or not to activate hover-info.
            annotation_label (string); Label for new annotations.
            show_annotations_only (bool): Whether or not to only show annotations.
            hide_bad_channels (int): Num clicks on hide-bad-channels-button button.
            highlight_model_channels (int): Num clicks on highlight-model-channels-button button.
            current_selected_bad_channels (list): List containing names of currently selected bad channels.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(plotly.graph_objs.Figure, int): Updated EEG plot.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        patched_fig = Patch()

        if globals.plotting_data:
            # If re-running model, keep current annotations and bad channels
            if 'rerun-model-button' in trigger and run_model_bool:

                globals.model_raw.info['bads'] = current_selected_bad_channels

                print('Running model...')
                run_model_output, run_model_channel_names, run_model_sample_rate, run_model_description = run_model(globals.model_raw, globals.viewing_raw)

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

                    model_annotations = confidence_intervals(run_model_output, model_threshold, 1, model_timestep)
                    for interval_index, interval in enumerate(model_annotations):
                        model_annotations[interval_index] = (interval[0], interval[1], run_model_description)

                    all_annotations = globals.marked_annotations + model_annotations
                    all_annotations, _ = merge_intervals(all_annotations)

                    globals.marked_annotations = all_annotations

                    annotations_to_raw(globals.raw, globals.marked_annotations)
                    annotations_to_raw(globals.viewing_raw, globals.marked_annotations)
                
                if not globals.plotting_data['model']:
                    globals.plotting_data['model'].append({})

                globals.plotting_data['model'][-1]['model_data'] = run_model_output
                globals.plotting_data['model'][-1]['model_channels'] = run_model_channel_names
                globals.plotting_data['model'][-1]['model_timescale'] = np.linspace(0, globals.plotting_data['EEG']['recording_length'], num=run_model_output.shape[0])
                globals.plotting_data['model'][-1]['offset_model_data'] = [-((2 + len(globals.plotting_data['model']) - 1) * (globals.plotting_data['plot']['offset_factor'])) for i in range(len(globals.plotting_data['model'][-1]['model_timescale']))]
                
                # current_fig['data'][-1]['marker']['color'] = run_model_output
                
                # current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'] = globals.plotting_data['EEG']['highlighted_channel_colors']
                # current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'] = globals.plotting_data['EEG']['default_channel_colors']

                updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0), reorder_channels)

                return updated_fig

            if 'reset-models' in trigger:
                if len(globals.plotting_data['model']) > 0:
                    patched_fig['data'] = current_fig['data'][:-len(globals.plotting_data['model'])]
                    patched_fig['layout']['yaxis']['tickvals'] = current_fig['layout']['yaxis']['tickvals'][len(globals.plotting_data['model']):]
                    patched_fig['layout']['yaxis']['ticktext'] = current_fig['layout']['yaxis']['ticktext'][len(globals.plotting_data['model']):]
                    patched_fig['layout']['yaxis']['range'] = ((-2 * (c.DEFAULT_Y_AXIS_OFFSET)), ((len(globals.plotting_data['EEG']['channel_names']) + 1) * (c.DEFAULT_Y_AXIS_OFFSET)))

                    globals.plotting_data['plot']['y_ticks'] = globals.plotting_data['plot']['y_ticks'][len(globals.plotting_data['model']):]
                    globals.plotting_data['plot']['y_tick_labels'] = globals.plotting_data['plot']['y_tick_labels'][len(globals.plotting_data['model']):]
                    del globals.plotting_data['model'][:]

                    # updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only, skip_hoverinfo)

                    return patched_fig
                else:
                    raise PreventUpdate

            # if 'run-model' in trigger and not run_model_bool:
            #     if len(globals.plotting_data['model']) > 0:
            #         del globals.plotting_data['model'][-1]

            #         patched_fig['data'] = current_fig['data'][:-1]

            #         # updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only, skip_hoverinfo)

            #         return patched_fig
            #     else:
            #         raise PreventUpdate

            if 'model-threshold' in trigger or 'annotate-model' in trigger:
                all_model_annotations = []
                
                if model_annotate:
                    for model in globals.plotting_data['model']:
                        model_timestep = model['model_timescale'][1]

                        output_intervals = confidence_intervals(model['model_data'], model_threshold, 1, model_timestep)
                        for interval_index, interval in enumerate(output_intervals):
                            output_intervals[interval_index] = (interval[0], interval[1], 'bad_artifact_model')
                        all_model_annotations = all_model_annotations + output_intervals

                remaining_annotations = [annotation for annotation in globals.marked_annotations if annotation[2] != 'bad_artifact_model']

                merged_annotations, _ = merge_intervals(all_model_annotations + remaining_annotations)

                globals.marked_annotations = merged_annotations

                annotations_to_raw(globals.raw, globals.marked_annotations)
                annotations_to_raw(globals.viewing_raw, globals.marked_annotations)

                if show_annotations_only and len(globals.marked_annotations) > 0:
                    globals.current_plot_index = 0

                    globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                    globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2

                    updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label,  show_annotation_labels, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0), reorder_channels)

                    return updated_fig
                else:
                    patched_fig['layout']['shapes'] = []
                    for annotation in globals.marked_annotations:
                        patched_fig['layout']['shapes'].append({
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

                    return patched_fig

        raise PreventUpdate
