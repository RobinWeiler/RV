from dash import Patch, ALL
from dash_extensions.enrich import Serverside, Output, Input, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

import numpy as np
import torch

import RV.constants as c
from RV.callbacks.utils.annotation_utils import get_annotation_label_radioitem, merge_annotations, annotations_to_raw
from RV.callbacks.utils.bad_channels_utils import bad_channel_disagrees


def register_model_callbacks():
    @callback(
        Output('RV-model-data', 'data'),
        [
            Input('RV-model-file-selection-dropdown', 'value'),
            Input('RV-run-model', 'value')
        ],
        State('RV-model-data', 'data'),
        prevent_initial_call=True
    )
    def update_model_data(selected_model_files, run_model, model_data):
        """Add model predictions of selected model-files to RV-model-data.
        If RV-run-model is active, add RV-model-data entry with annotation description.
        Predictions will be calculated once plot button is clicked (visualization_callbacks.py).
        """
        trigger = ctx.triggered_id
        # print(trigger)

        if 'RV-model-file-selection-dropdown' in trigger and selected_model_files:
            for model_index, file_path in enumerate(selected_model_files):
                if '.txt' in file_path:
                    model_predictions = np.loadtxt(file_path)
                elif '.pt' in file_path:
                    model_predictions = torch.load(file_path)
                elif '.npy' in file_path:
                    model_predictions = np.load(file_path)
                else:
                    raise Exception('Unknown file type. Currently, only .txt, .pt, and .npy files are supported for model-prediction files.')

                model_data[f'M{model_index + 1}'] = {'predictions': model_predictions, 'channel_names': None, 'annotation_description': f'annotation_M{model_index + 1}'}

        elif 'RV-run-model' in trigger and run_model:
            model_data['M0'] = {'annotation_description': 'annotation_M0'}

        return model_data

    @callback(
        Output('RV-model-buttons', 'style'),
        Input('RV-model-data', 'data'),
        prevent_initial_call=True
    )
    def hide_model_buttons(model_data):
        """Hide model buttons in menu bar if no model predictions are used.
        """
        if len(model_data) > 0:
            return {}
        else:
            return {'display': 'none'}

    @callback(
        Output('RV-highlight-model-channels-button', 'disabled'),
        Input('RV-model-data', 'data'),
        prevent_initial_call=True
    )
    def disable_highlight_model_channels_button(model_data):
        """Enable RV-highlight-model-channels-button if any model in RV-model-data provides list of channel names predictions are based on.
        Disable otherwise.
        """
        for model in model_data.values():
            if 'channel_names' in model.keys() and model['channel_names']:
                return False

        return True

    @callback(
        Output('RV-model-threshold-input', 'disabled'),
        Input('RV-annotate-model', 'value'),
        prevent_initial_call=True
    )
    def disable_model_threshold_input(model_annotate):
        """Disable model-threshold input if RV-annotate-model is not active.
        """
        return not model_annotate

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-main-graph', 'figure', allow_duplicate=True),
            Output('RV-refresh-annotations-button', 'n_clicks', allow_duplicate=True),
            Output('RV-annotation-label', 'options', allow_duplicate=True)
        ],
        [
            Input('RV-annotate-model', 'value'),
            Input('RV-model-threshold-input', 'value')
        ],
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-model-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-show-annotation-labels', 'value'),
            State('RV-annotations-only-mode', 'value'),
            State('RV-annotation-label', 'options'),
            State({'type': 'color-dropdown', 'label': ALL}, 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'id'),
        ],
        prevent_initial_call=True
    )
    def mark_model_annotations(model_annotate, model_threshold,
                               raw,
                               plotting_data,
                               model_data,
                               resampler,
                               show_annotation_labels,
                               annotations_only_mode,
                               annotation_label_options,
                               annotation_colors, annotation_colors_ids
    ):
        """Marks annotations according to model_threshold and all model predictions in RV-model-data.
        Also adds new annotation-label option for model annotations.
        Triggered when RV-annotate-model is activated or RV-model-threshold-input changes.
        """
        if resampler is None:
            raise PreventUpdate

        patched_fig = Patch()

        if model_annotate and model_threshold:
            current_annotation_labels = [annotation_label['value'] for annotation_label in annotation_label_options]

            all_model_annotations = []
            for model_name, model in model_data.items():
                if model['annotation_description'] not in current_annotation_labels:
                    annotation_option, annotation_color = get_annotation_label_radioitem(model['annotation_description'])
                    annotation_label_options.append(annotation_option)

                if 'predictions' in model.keys():
                    model_prediction_times = np.linspace(0, plotting_data['recording_length'], len(model['predictions']))
                    predictions_greater_equal_threshold = np.greater_equal(model['predictions'], model_threshold)

                    model_annotations = []
                    model_annotation_onset = None
                    for prediction_index, prediction in enumerate(predictions_greater_equal_threshold):
                        if prediction:  # if prediction >= model_threshold
                            if model_annotation_onset is None:  # if new annotation starts, record onset (else next prediction)
                                prediction_time = model_prediction_times[prediction_index]
                                model_annotation_onset = prediction_time
                        elif model_annotation_onset != None:  # if prediction < model_threshold and previous prediction >= model_threshold, create annotation
                            prediction_time = model_prediction_times[prediction_index]
                            model_annotations.append((model_annotation_onset, prediction_time - model_annotation_onset, model['annotation_description']))
                            model_annotation_onset = None
                        else:
                            continue

                    # if last model annotation goes until end of recording
                    if model_annotation_onset is not None:
                        model_annotations.append((model_annotation_onset, model_prediction_times[-1] - model_annotation_onset, model['annotation_description']))

                    all_model_annotations += model_annotations

            loaded_annotations = []
            for annotation_index in range(len(raw.annotations)):
                # Skip annotations with label model['annotation_description'] to replace those
                if raw.annotations.description[annotation_index] != model['annotation_description']:
                    loaded_annotations.append((raw.annotations.onset[annotation_index], raw.annotations.duration[annotation_index], raw.annotations.description[annotation_index]))

            merged_annotations, _ = merge_annotations(loaded_annotations + all_model_annotations)
            # print(f'Current annotations: {merged_annotations}')

            raw = annotations_to_raw(merged_annotations, raw)

            patched_fig['layout']['shapes'] = []
            for annotation_index in range(len(raw.annotations)):
                for i, dropdown in enumerate(annotation_colors_ids):
                    if dropdown['label'] == raw.annotations.description[annotation_index]:
                        annotation_color = annotation_colors[i]
                        break

                patched_fig['layout']['shapes'].append({
                    'editable': True,
                    'fillcolor': annotation_color if annotation_color != 'hide' else 'red',
                    'label': {'text': raw.annotations.description[annotation_index], 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}} if show_annotation_labels else {},
                    'layer': 'below',
                    'line': {'width': 0},
                    'name': raw.annotations.description[annotation_index],
                    'opacity': 0.5,
                    'type': 'rect',
                    'visible': True if annotation_color != 'hide' else False,
                    'x0': raw.annotations.onset[annotation_index],
                    'x1': raw.annotations.onset[annotation_index] + raw.annotations.duration[annotation_index],
                    'xref': 'x',
                    'y0': resampler['layout']['yaxis']['range'][0],
                    'y1': resampler['layout']['yaxis']['range'][1],
                    'yref': 'y'
                })

            if annotations_only_mode:
                refresh_needed = 1
            else:
                refresh_needed = no_update

            return Serverside(raw), patched_fig, refresh_needed, annotation_label_options

        raise PreventUpdate

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True),
            Output('RV-main-graph-resampler', 'data', allow_duplicate=True),
            Output('RV-channel-selection-graph', 'figure', allow_duplicate=True)
        ],
        Input('RV-highlight-model-channels-button', 'n_clicks'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-model-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-bad-channels-dropdown', 'value'),
            State('RV-channel-selection-graph', 'figure'),
        ],
        prevent_initial_call=True
    )
    def highlight_model_channels(highlight_model_channels, raw, plotting_data, model_data, resampler, selected_bad_channels, channel_selection_fig):
        """Highlight channels used for model predictions in RV-main-graph (and RV-channel-selection-graph).
        """
        patched_fig = Patch()

        if channel_selection_fig != None:
            patched_channel_fig = Patch()
        else:
            patched_channel_fig = no_update

        for model in model_data.values():
            if 'channel_names' in model.keys() and model['channel_names']:
                for channel_name in model['channel_names']:
                    trace_index = plotting_data['selected_channels'].index(channel_name)
                    if channel_selection_fig != None:
                        channel_index = raw.ch_names.index(channel_name)

                    if highlight_model_channels % 2 != 0:
                        patched_fig['data'][trace_index]['marker']['color'] = c.MODEL_CHANNEL_COLOR
                        resampler['data'][trace_index]['marker']['color'] = c.MODEL_CHANNEL_COLOR
                        if channel_selection_fig != None:
                            patched_channel_fig['data'][channel_index]['marker']['color'] = c.MODEL_CHANNEL_COLOR

                    else:
                        if channel_name in selected_bad_channels:
                            if bad_channel_disagrees(channel_name, plotting_data['bad_channels']):
                                channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
                            else:
                                channel_color = c.BAD_CHANNEL_COLOR
                        else:
                            channel_color = 'black'

                        patched_fig['data'][trace_index]['marker']['color'] = channel_color
                        resampler['data'][trace_index]['marker']['color'] = channel_color
                        if channel_selection_fig != None:
                            patched_channel_fig['data'][channel_index]['marker']['color'] = channel_color

        return patched_fig, Serverside(resampler), patched_channel_fig
