import json
import re

import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

import numpy as np

from helperfunctions.annotation_helperfunctions import merge_intervals, annotations_to_raw, get_annotations
from helperfunctions.loading_helperfunctions import parse_annotation_file
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.saving_helperfunctions import quick_save

import globals


def register_annotation_callbacks(app):
    
    # Toggle annotation-settings modal
    @app.callback(
        Output("modal-annotation-settings", "is_open"),
        [Input("open-annotation-settings", "n_clicks"), Input("close-annotation-settings", "n_clicks")],
        [State("modal-annotation-settings", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_annotation_settings_modal(open_annotation_settings, close_annotation_settings, is_open):
        """Opens or closes help modal based on relevant button clicks.

        Args:
            open_annotation_settings (int): Num clicks on open-annotation-settings button.
            close_annotation_settings (int): Num clicks on close-annotation-settings button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_annotation_settings, close_annotation_settings], is_open)

    # Annotation through dragging mouse across intervals callback
    @app.callback(
        [Output('segment-slider', 'max', allow_duplicate=True), Output('segment-slider', 'step', allow_duplicate=True), Output('segment-slider', 'marks', allow_duplicate=True), Output('segment-slider', 'value', allow_duplicate=True)],
        Input('EEG-graph', 'relayoutData'),
        [State('annotation-label', 'value'), State('show-annotations-only', 'value'), State('EEG-graph', 'figure')],
        prevent_initial_call=True
    )
    def _make_annotation(relayoutData, annotation_label, show_annotations_only, current_fig):
        """Saves annotations when new ones are made or old ones are moved/deleted. Triggers when user zooms, pans, and draws on plot.

        Args:
            relayoutData (dict): Data from latest relayout event.
            annotation_label (string); Label for new annotations.
        """
        # print(relayoutData)
        if relayoutData:
            # Annotation was added/removed/moved/resized
            if any('shapes' in key for key in relayoutData.keys()):
                globals.marked_annotations[:] = []

                for shape in current_fig['layout']['shapes']:
                    if shape['type'] == 'rect':
                        x0 = np.round(shape['x0'], 3)
                        x1 = np.round(shape['x1'], 3)

                        if x0 < x1:
                            annotation_start = x0
                            annotation_end = x1
                        else:
                            annotation_start = x1
                            annotation_end = x0
                        
                        globals.marked_annotations.append((annotation_start, annotation_end, annotation_label))

                globals.marked_annotations = merge_intervals(globals.marked_annotations)

                globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)

                quick_save(globals.raw)
                print(globals.marked_annotations)

                if show_annotations_only and len(globals.marked_annotations) > 0:
                    num_segments = int(len(globals.marked_annotations) - 1)
                    marks = {i: '{}'.format(i) for i in range(num_segments + 1)}

                    return num_segments, 1, marks, globals.current_plot_index

        raise PreventUpdate

    # Add/remove/rename annotation label
    @app.callback(
        [Output('annotation-label', 'options'), Output('new-annotation-label', 'value'), Output('annotation-label', 'value')],
        [Input('data-file', 'children'), Input('model-output-files', 'children'), Input('new-annotation-label', 'value'), Input('remove-annotation-label', 'n_clicks'), Input('rename-annotation-label', 'n_clicks')],
        [State('annotation-label', 'options'), State('annotation-label', 'value'), State('renamed-annotation-label', 'value')],
        prevent_initial_call=True
    )
    def _add_annotation_label(current_file_name, loaded_annotation_files, new_annotation_label, remove_annotations_button, rename_annotation_label_button, annotation_labels, current_annotation_label, renamed_annotation_label):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)

        if 'data-file' in trigger:
            if globals.raw:
                loaded_annotations = get_annotations(globals.raw)

                for annotation in loaded_annotations:
                    if annotation[2] not in globals.annotation_label_colors.keys():
                        globals.annotation_label_colors[annotation[2]] = 'red'
                        annotation_labels.append({'label': '{}'.format(annotation[2]), 'value': '{}'.format(annotation[2])})

        elif 'model-output-files' in trigger:
            for file_name in loaded_annotation_files:
                if '.csv' in file_name:
                    loaded_annotations = parse_annotation_file(file_name)

                    for annotation in loaded_annotations:
                        if annotation[2] not in globals.annotation_label_colors.keys():
                            annotation_labels.append({'label': '{}'.format(annotation[2]), 'value': '{}'.format(annotation[2])})
                            globals.annotation_label_colors[annotation[2]] = 'red'

        elif 'remove-annotation-label' in trigger and len(annotation_labels) > 1:
            remove_annotation_label = current_annotation_label
            current_annotation_label = annotation_labels[-1]['value']

            annotation_labels.remove({'label': remove_annotation_label, 'value': remove_annotation_label})
            globals.annotation_label_colors.pop(remove_annotation_label)

            globals.marked_annotations = [annotation for annotation in globals.marked_annotations if annotation[2] != remove_annotation_label]
            globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)
            quick_save(globals.raw)

        elif 'rename-annotation-label' in trigger and len(annotation_labels) > 0:
            rename_annotation_label_index = next((index for (index, d) in enumerate(annotation_labels) if d["label"] == current_annotation_label), None)

            annotation_labels[rename_annotation_label_index] = {'label': renamed_annotation_label, 'value': renamed_annotation_label}
            globals.annotation_label_colors[renamed_annotation_label] = globals.annotation_label_colors.pop(current_annotation_label)
            if current_annotation_label == globals.model_annotation_label:
                globals.model_annotation_label = renamed_annotation_label

            globals.marked_annotations = [(annotation[0], annotation[1], renamed_annotation_label) if annotation[2] == current_annotation_label else annotation for annotation in globals.marked_annotations]
            globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)
            quick_save(globals.raw)

            current_annotation_label = renamed_annotation_label

        elif 'new-annotation-label' in trigger and new_annotation_label not in globals.annotation_label_colors.keys():
            annotation_labels.append({'label': '{}'.format(new_annotation_label), 'value': '{}'.format(new_annotation_label)})
            globals.annotation_label_colors[new_annotation_label] = 'red'

        return annotation_labels, '', current_annotation_label

    # Change username
    @app.callback(
        Output('username-dummy', 'children', allow_duplicate=True),
        Input('username', 'value'),
        prevent_initial_call=True
    )
    def _change_username(username):
        print(username)
        if globals.raw and globals.username != username:
            globals.username = username
            globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)
            quick_save(globals.raw)
        else:
            globals.username = username

    # Switch to current annotation-label color
    @app.callback(
        Output('annotation-label-color', 'value'),
        Input('annotation-label', 'value'),
        # State('annotation-label', 'options'),
        prevent_initial_call=True
    )
    def _switch_annotation_label_color(current_annotation_label):
        color = globals.annotation_label_colors[current_annotation_label]
        
        return color

    # Choose annotation-label color
    @app.callback(
        Output('chosen-annotation-color', 'children'),
        Input('annotation-label-color', 'value'),
        State('annotation-label', 'value'),
        prevent_initial_call=True
    )
    def _choose_annotation_label_color(current_annotation_label_color, current_annotation_label):
        globals.annotation_label_colors[current_annotation_label] = current_annotation_label_color

    # Update plot when annotation-label or annotation-label-color is changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        [Input('annotation-label', 'value'), Input('annotation-label-color', 'value')],
        State('EEG-graph', 'figure'),
        prevent_initial_call=True
    )
    def _update_annotations(annotation_label, annotation_label_color, current_fig):
        """Updates annotations.

        Args:
            annotation_label (string); Label for new annotations.
            annotation_label_color (dict); Color for new annotations.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(plotly.graph_objs.Figure, int): Updated EEG plot.
        """
        if globals.plotting_data:
            patched_fig = Patch()

            patched_fig['layout']['newshape']['fillcolor'] = annotation_label_color
            
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

    # Toggle rename-annotation-labels modal
    @app.callback(
        Output("modal-rename-annotation-label", "is_open"),
        [Input("rename-annotation-label-modal-button", "n_clicks"), Input("cancel-rename-annotation-label-button", "n_clicks"), Input('rename-annotation-label', 'n_clicks')],
        [State("modal-rename-annotation-label", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_save_modal(open_rename_annotation_label, close_rename_annotation_label, rename_annotation_label, is_open):
        """Opens or closes rename-annotation-label modal based on relevant button clicks.

        Args:
            open_rename_annotation_label (int): Num clicks on rename-annotation-label-modal-button button.
            close_rename_annotation_label (int): Num clicks on cancel-rename-annotation-label-button button.
            rename_annotation_label (int): Num clicks on rename-annotation-label button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_rename_annotation_label, close_rename_annotation_label, rename_annotation_label], is_open)

    # Toggle remove-annotation-labels modal
    @app.callback(
        Output("modal-remove-annotation-label", "is_open"),
        [Input("remove-annotation-label-modal-button", "n_clicks"), Input("cancel-remove-annotation-label-button", "n_clicks"), Input('remove-annotation-label', 'n_clicks')],
        [State("modal-remove-annotation-label", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_save_modal(open_remove_annotation_label, close_remove_annotation_label, remove_annotation_label, is_open):
        """Opens or closes remove-annotation-label modal based on relevant button clicks.

        Args:
            open_remove_annotation_label (int): Num clicks on remove-annotation-label-modal-button button.
            close_remove_annotation_label (int): Num clicks on cancel-remove-annotation-label-button button.
            remove_annotation_label (int): Num clicks on remove-annotation-label button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_remove_annotation_label, close_remove_annotation_label, remove_annotation_label], is_open)
