import math

from dash import Patch, ALL
from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback, ctx, no_update
from dash.exceptions import PreventUpdate

import mne
import pandas as pd
import numpy as np

import RV.constants as c
from RV.callbacks.utils.annotation_utils import get_annotation_label_radioitem, merge_annotations, annotations_to_raw


def register_annotation_callbacks():
    # Activate annotation mode when RV-mark-annotations-button is clicked
    clientside_callback(
        """
            function(n_clicks) {
                document.querySelector("a[data-val='drawrect']").click()
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-mark-annotations-button', 'n_clicks'),
        Input('RV-mark-annotations-button', 'n_clicks'),
        prevent_initial_call=True
    )

    # Delete currently selected annotation when RV-delete-annotation-button is clicked
    clientside_callback(
        """
            function(n_clicks) {
                document.querySelector("a[data-title='Erase active shape']").click()
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-delete-annotation-button', 'n_clicks'),
        Input('RV-delete-annotation-button', 'n_clicks'),
        prevent_initial_call=True
    )

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-annotation-overview-graph', 'figure', allow_duplicate=True),
            Output('RV-refresh-annotations-button', 'n_clicks', allow_duplicate=True),
        ],
        Input('RV-main-graph', 'relayoutData'),
        [
            State('RV-raw', 'data'), State('RV-plotting-data', 'data'),
            State('RV-main-graph', 'figure'),
            State('RV-annotation-overview-graph', 'figure'),
            State('RV-annotations-only-mode', 'value'),
            State('RV-segment-size-input', 'value')
        ],
        prevent_initial_call=True
    )
    def update_annotations(relayout_data, raw, plotting_data, current_fig, current_annotation_fig, annotations_only_mode, segment_size):
        """Saves annotations to mne.io.Raw object when new ones are made or old ones are moved/deleted.
        Also synchronizes RV-annotation-overview-graph.
        Triggered when relayout event occurs but logic is only executed if user drew on plot or deleted an annotation.
        """
        if not any('shapes' in key for key in relayout_data.keys()):
            # relayout events not related to shapes are handled by different callbacks
            raise PreventUpdate

        if annotations_only_mode:
            refresh_needed = 1
        else:
            refresh_needed = no_update

        patched_annotation_fig = Patch()

        current_shapes = current_fig['layout']['shapes']
        for shape in current_shapes:
            # Change these attributes for annotations in RV-annotation-overview-graph
            shape['editable'] = False
            shape['label'] = {}
            shape['layer'] = 'above'
        patched_annotation_fig['layout']['shapes'] = current_shapes

        if segment_size:
            # Highlight currently plotted segment with dark rectangles left and right
            patched_annotation_fig['layout']['shapes'].append({
                'editable': False,
                'fillcolor': 'black',
                'layer': 'above',
                'line': {'width': 0},
                'opacity': 0.3,
                'type': 'rect',
                'x0': current_annotation_fig['layout']['xaxis']['range'][0],
                'x1': current_fig['layout']['xaxis']['range'][0],
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y'
            })
            patched_annotation_fig['layout']['shapes'].append({
                'editable': False,
                'fillcolor': 'black',
                'layer': 'above',
                'line': {'width': 0},
                'opacity': 0.3,
                'type': 'rect',
                'x0': current_fig['layout']['xaxis']['range'][1],
                'x1': current_annotation_fig['layout']['xaxis']['range'][1],
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y'
            })

        # Get list of (annotation_onset, annotation_duration, annotation_description)
        marked_annotations = []
        for shape in current_shapes:
            if shape['type'] == 'rect':
                x0 = shape['x0']
                x1 = shape['x1']
                annotation_label = shape['name']

                annotation_start = x0 if x0 < x1 else x1
                annotation_end = x1 if x0 < x1 else x0

                if annotation_start < 0:
                    annotation_start = 0
                    refresh_needed = 1
                if annotation_end > plotting_data['recording_length']:
                    annotation_end = plotting_data['recording_length']
                    refresh_needed = 1

                marked_annotations.append((round(annotation_start, 3), round(annotation_end - annotation_start, 3), annotation_label))

        if len(marked_annotations) > 1:
            marked_annotations, merge_happened = merge_annotations(marked_annotations)
            if merge_happened:
                refresh_needed = 1

        raw = annotations_to_raw(marked_annotations, raw)
        print(f'Current annotations: {marked_annotations}')

        return Serverside(raw), patched_annotation_fig, refresh_needed

    @callback(
        Output('RV-main-graph', 'figure', allow_duplicate=True),  # Output('RV-main-graph-resampler', 'data', allow_duplicate=True)
        [
            Input('RV-refresh-annotations-button', 'n_clicks'),
            Input('RV-show-annotation-labels', 'value')
        ],
        [
            State('RV-raw', 'data'), 
            State('RV-main-graph', 'figure'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-annotation-label', 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'id')
        ],
        prevent_initial_call=True
    )
    def refresh_annotations(refresh_annotations, show_annotation_labels, raw, current_fig, resampler, annotation_label, annotation_colors, annotation_colors_ids):
        """Refreshes RV-main-graph with updated annotations.
        Triggered when hidden RV-refresh-annotations-button is pressed or RV-show-annotation-labels is enabled/disabled.
        """
        if resampler is None:
            raise PreventUpdate

        trigger = ctx.triggered_id
        # print(trigger)

        patched_fig = Patch()

        if 'RV-show-annotation-labels' in trigger:
            if show_annotation_labels:
                new_yaxis_end = current_fig['layout']['yaxis']['range'][1] + (4 * c.DEFAULT_Y_AXIS_OFFSET)
            else:
                new_yaxis_end = current_fig['layout']['yaxis']['range'][1] - (4 * c.DEFAULT_Y_AXIS_OFFSET)

            patched_fig['layout']['yaxis']['range'][1] = new_yaxis_end
            patched_fig['layout']['newshape']['label'] = {'text': annotation_label, 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}} if show_annotation_labels else {}

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
                'y0': current_fig['layout']['yaxis']['range'][0],
                'y1': current_fig['layout']['yaxis']['range'][1] if not 'show-annotation-labels' in trigger else new_yaxis_end,
                'yref': 'y'
            })

        return patched_fig

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-refresh-annotations-button', 'n_clicks', allow_duplicate=True)
        ],
        Input('RV-annotation-file-selection-dropdown', 'value'),
        State('RV-raw', 'data'),
        prevent_initial_call=True
    )
    def load_annotation_files(selected_annotation_files, raw):
        """Loads annotations from annotation files selected in RV-annotation-file-selection-dropdown into mne.io.Raw object.
        Currently supported file formats are .csv and .fif.
        """
        if raw is None:
            raise PreventUpdate

        if selected_annotation_files:
            for file_path in selected_annotation_files:
                if '.csv' in file_path:
                    loaded_annotations_df = pd.read_csv(file_path)
                    
                    annotation_onsets = loaded_annotations_df['onset'].tolist()
                    annotation_durations = loaded_annotations_df['duration'].tolist()
                    annotation_descriptions = loaded_annotations_df['description'].tolist()

                    for annotation_index in range(len(annotation_onsets)):
                        raw.annotations.append(annotation_onsets[annotation_index], annotation_durations[annotation_index], annotation_descriptions[annotation_index])

                elif '.fif' in file_path:
                    loaded_annotations = mne.io.read_raw(file_path).annotations

                    for annotation_index in range(len(loaded_annotations.onset)):
                        raw.annotations.append(loaded_annotations.onset[annotation_index], loaded_annotations.duration[annotation_index], loaded_annotations.description[annotation_index])

                else:
                    raise Exception('Unknown file type. Currently, only .csv and .fif files are supported for annotation files.')
        
            return Serverside(raw), 1

        raise PreventUpdate

    @callback(
        [
            Output('RV-annotation-label', 'options', allow_duplicate=True),
            Output('RV-new-annotation-label-input', 'value', allow_duplicate=True)
        ],
        [
            Input('RV-annotation-file-selection-dropdown', 'value'),
            Input('RV-new-annotation-label-input', 'value'),
        ],
        [
            State('RV-annotation-label', 'options'),
        ],
        prevent_initial_call=True
    )
    def add_annotation_labels(selected_annotation_files, new_annotation_label, annotation_label_options):
        """Adds annotation-label options. Triggered when annotation files are selected or a new annotation label is added.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        current_annotation_labels = [annotation_label['value'] for annotation_label in annotation_label_options]

        if 'RV-annotation-file-selection-dropdown' in trigger:
            if selected_annotation_files:
                for file_path in selected_annotation_files:
                    if '.csv' in file_path:
                        selected_annotations = pd.read_csv(file_path)

                        # get all descriptions and remove duplicates
                        selected_annotation_labels = list(set(selected_annotations['description'].tolist()))

                    elif '.fif' in file_path:
                        selected_annotations = mne.io.read_raw(file_path).annotations

                        # get all descriptions and remove duplicates
                        selected_annotation_labels = list(set(selected_annotations.description))

                    else:
                        raise Exception('Unknown file type. Currently, only .csv and .fif files are supported for annotation files.')

                    for annotation_label in selected_annotation_labels:
                        if annotation_label not in current_annotation_labels:
                            annotation_label_options.append(get_annotation_label_radioitem(annotation_label)[0])

                return annotation_label_options, no_update
            else:
                raise PreventUpdate

        elif 'RV-new-annotation-label-input' in trigger:
            if new_annotation_label not in current_annotation_labels:
                annotation_label_options.append(get_annotation_label_radioitem(new_annotation_label)[0])
            else:
                raise PreventUpdate

            return annotation_label_options, ''

        raise PreventUpdate

    @callback(
        Output({'type': 'open-button', 'modal': 'RV-remove-annotation-label'}, 'disabled'),
        Input('RV-annotation-label', 'options'),
        prevent_initial_call=True
    )
    def disable_remove_annotation_label_button(annotation_labels):
        """Enables {'type': 'open-button', 'modal': 'RV-remove-annotation-label'} if there is more than 1 annotation label.
        Disables otherwise.
        """
        if len(annotation_labels) > 1:
            return False

        return True

    @callback(
        Output('RV-renamed-annotation-label', 'value', allow_duplicate=True),
        Input({'type': 'process-button', 'modal': 'RV-rename-annotation-label'}, 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_renamed_annotation_label(rename_annotation_label):
        """Clears RV-renamed-annotation-label. Triggered by {'type': 'process-button', 'modal': 'RV-rename-annotation-label'}.
        """
        if rename_annotation_label:
            return ''

        raise PreventUpdate

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-model-data', 'data', allow_duplicate=True),
            Output('RV-annotation-label', 'options', allow_duplicate=True),
            Output('RV-annotation-label', 'value', allow_duplicate=True),
            Output('RV-refresh-annotations-button', 'n_clicks', allow_duplicate=True)
        ],
        [
            Input({'type': 'process-button', 'modal': 'RV-remove-annotation-label'}, 'n_clicks'),
            Input({'type': 'process-button', 'modal': 'RV-rename-annotation-label'}, 'n_clicks'),
        ],
        [
            State('RV-raw', 'data'),
            State('RV-model-data', 'data'),
            State('RV-annotation-label', 'options'), State('RV-annotation-label', 'value'),
            State('RV-renamed-annotation-label', 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'value'), State({'type': 'color-dropdown', 'label': ALL}, 'id'),
        ],
        prevent_initial_call=True
    )
    def update_annotation_labels(remove_annotations_button, rename_annotation_label_button,
                                 raw,
                                 model_data,
                                 annotation_label_options, selected_annotation_label,
                                 new_annotation_label,
                                 annotation_colors, annotation_colors_ids
    ):
        """Removes or renames annotations of selected label.
        Updates mne.io.Raw object and RV-annotation-label options.
        Also clicks RV-refresh-annotations-button (if necessary).
        Triggered when annotation label is removed or renamed (using respective buttons).
        """
        trigger = ctx.triggered_id
        # print(trigger)

        current_annotation_labels = [annotation_label['value'] for annotation_label in annotation_label_options]

        if 'RV-remove-annotation-label' in trigger['modal']:
            if (raw != None) and (selected_annotation_label in raw.annotations.description):
                refresh_annotations = 1

                remove_indices = []
                for annotation_index, annotation_label in enumerate(raw.annotations.description):
                    if annotation_label == selected_annotation_label:
                        remove_indices.append(annotation_index)

                raw.annotations.delete(remove_indices)
            else:
                refresh_annotations = no_update

            annotation_label_options.pop(current_annotation_labels.index(selected_annotation_label))
            selected_annotation_label = annotation_label_options[0]['value']

            model_data = no_update

        elif 'RV-rename-annotation-label' in trigger['modal']:
            if (raw != None) and (selected_annotation_label in raw.annotations.description):
                refresh_annotations = 1

                raw.set_annotations(raw.annotations.rename({f'{selected_annotation_label}': f'{new_annotation_label}'}))
            else:
                refresh_annotations = no_update

            if model_data != None:
                for model_name, model in model_data.items():
                    if selected_annotation_label == model['annotation_description']:
                        model['annotation_description'] = new_annotation_label
            else:
                model_data = no_update

            for index, dropdown in enumerate(annotation_colors_ids):
                if dropdown['label'] == selected_annotation_label:
                    annotation_color = annotation_colors[index]
                    break

            annotation_label_options[current_annotation_labels.index(selected_annotation_label)] = get_annotation_label_radioitem(new_annotation_label, annotation_color)[0]
            selected_annotation_label = new_annotation_label

        return Serverside(raw), model_data, annotation_label_options, selected_annotation_label, refresh_annotations

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True),
            Output('RV-refresh-annotations-button', 'n_clicks', allow_duplicate=True)
        ],
        [
            Input('RV-annotation-label', 'value'),
            Input({'type': 'color-dropdown', 'label': ALL}, 'value')
        ],
        [
            State('RV-raw', 'data'), State('RV-main-graph-resampler', 'data'),
            State({'type': 'color-dropdown', 'label': ALL}, 'id'),
            State('RV-show-annotation-labels', 'value')
        ],
        prevent_initial_call=True
    )
    def update_annotation_drawing(selected_annotation_label, annotation_colors, raw, resampler, annotation_colors_ids, show_annotation_labels):
        """Updates attributes of annotation-drawing when different annotation label is selected or annotation-label color is changed.
        """
        if resampler is None:
            raise PreventUpdate

        trigger = ctx.triggered_id
        # print(trigger)

        patched_fig = Patch()
        refresh_annotations = no_update

        # Get annotation color of selected label
        for index, dropdown in enumerate(annotation_colors_ids):
            if dropdown['label'] == selected_annotation_label:
                annotation_color = annotation_colors[index]
                break

        patched_fig['layout']['newshape']['fillcolor'] = annotation_color if annotation_color != 'hide' else 'red'
        patched_fig['layout']['newshape']['visible'] = True if annotation_color != 'hide' else False

        if 'RV-annotation-label' in trigger:
            patched_fig['layout']['newshape']['name'] = selected_annotation_label
            patched_fig['layout']['newshape']['label']['text'] = selected_annotation_label if show_annotation_labels else ''

        else:
            # If color was changed of existing annotation-label, redraw annotations
            if selected_annotation_label in raw.annotations.description:
                refresh_annotations = 1

        return patched_fig, refresh_annotations

    @callback(
        Output({'type': 'modal', 'modal': 'RV-settings'}, 'is_open', allow_duplicate=True),
        [
            Input({'type': 'modal', 'modal': 'RV-remove-annotation-label'}, 'is_open'),
            Input({'type': 'modal', 'modal': 'RV-rename-annotation-label'}, 'is_open')
        ],
        prevent_initial_call=True
    )
    def toggle_settings_modal_annotations(remove_annotation_label_is_open, rename_annotation_label_is_open):
        """Toggles RV-settings modal opposite to RV-remove-annotation-label modal and RV-rename-annotation-label modal.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        if 'remove' in trigger['modal']:
            return not remove_annotation_label_is_open
        elif 'rename' in trigger['modal']:
            return not rename_annotation_label_is_open

        raise PreventUpdate

    @callback(
        [
            Output('RV-segment-slider', 'max', allow_duplicate=True),
            Output('RV-segment-slider', 'marks', allow_duplicate=True),
            Output('RV-segment-slider', 'value', allow_duplicate=True),
            Output('RV-annotation-overview-graph', 'figure', allow_duplicate=True),
        ],
        [
            Input('RV-annotations-only-mode', 'value'),
            Input('RV-refresh-annotations-button', 'n_clicks'),
        ],
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-segment-slider', 'value'),
            State('RV-segment-size-input', 'value'),
            State('RV-annotation-overview-graph', 'figure'),
        ],
        prevent_initial_call=True
    )
    def update_segment_slider_annotations_only_mode(annotations_only_mode, refresh_annotations, raw, plotting_data, segment_slider, segment_size, current_annotation_fig):
        """Updates RV-segment-slider if annotations have changed and annotations-only mode is active.
        """
        if plotting_data is None:
            raise PreventUpdate

        trigger = ctx.triggered_id
        # print(trigger)

        if ('RV-refresh-annotations-button' in trigger) and not annotations_only_mode:
            raise PreventUpdate

        # if annotations-only mode was activated
        if annotations_only_mode:
            if len(raw.annotations) == 0:
                raise Exception('There are no artifacts to show.')
            num_segments = len(raw.annotations) - 1
            segment_slider_marks = {i: {'label': f'{i}'} for i in range(len(raw.annotations))}

            if 'RV-annotations-only-mode' in trigger:
                if (len(raw.annotations) == 1) or not segment_size:
                    segment_slider = 0
                else:
                    annotation_in_segment = False
                    # if there is an annotation in the currently viewed segment, set RV-segment-slider to this annotation
                    for annotation_index, annotation_onset in enumerate(raw.annotations.onset):
                        if (annotation_onset >= (segment_slider * segment_size)) and (annotation_onset < ((segment_slider + 1) * segment_size)):
                            segment_slider = annotation_index
                            annotation_in_segment = True
                            break

                    if not annotation_in_segment:
                        segment_slider = 0
            else:
                if segment_slider >= len(raw.annotations):
                    segment_slider = len(raw.annotations) - 1

        # if annotations-only mode was deactivated
        elif segment_size:
            num_segments = int(plotting_data['recording_length'] // segment_size)
            segment_slider_marks = {i: {'label': f'{i * segment_size}'} for i in range(num_segments + 1)}
            # Set RV-segment-slider to segment closest to the left of currently viewed annotation
            segment_slider = math.floor(raw.annotations.onset[segment_slider] / segment_size)
        else:
            num_segments = 0
            segment_slider_marks = {0: {'label': '0'}}
            segment_slider = 0

        # Update clickable points in RV-annotation-overview-graph with new RV-segment-slider parameters
        if annotations_only_mode:
            x_ticks = [(raw.annotations.onset[index] + (raw.annotations.duration[index] / 2)) for index in range(len(raw.annotations))]
            x_ticks = np.array(x_ticks)

            current_annotation_fig['data'][0]['x'] = x_ticks
            current_annotation_fig['data'][0]['y'] = np.repeat(0.9, len(x_ticks))
            current_annotation_fig['data'][0]['text'] = np.arange(len(x_ticks))
        elif segment_size:
            x_ticks = [(int(segment) * segment_size) for segment in segment_slider_marks.keys()]
            x_ticks = np.array(x_ticks)

            current_annotation_fig['data'][0]['x'] = x_ticks
            current_annotation_fig['data'][0]['y'] = np.repeat(0.9, len(x_ticks))
            current_annotation_fig['data'][0]['text'] = x_ticks
        else:
            current_annotation_fig['data'][0]['x'] = None
            current_annotation_fig['data'][0]['y'] = None
            current_annotation_fig['data'][0]['text'] = None

        return num_segments, segment_slider_marks, segment_slider, current_annotation_fig
