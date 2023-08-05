import json
import re

import dash
from dash.dependencies import Input, Output, State

import numpy as np

from helperfunctions.annotation_helperfunctions import merge_intervals, annotations_to_raw
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
        Output('relayout-data', 'children'),
        Input('EEG-graph', 'relayoutData'),
        [State('annotation-label', 'value'), State('show-annotations-only', 'value')],
        prevent_initial_call=True
    )
    def _make_annotation(relayoutData, annotation_label, show_annotations_only):
        """Saves annotations when new ones are made or old ones are moved/deleted. Triggers when user zooms, pans, and draws on plot.

        Args:
            relayoutData (dict): Data from latest relayout event.
            annotation_label (string); Label for new annotations.
        """
        # print(relayoutData)
        if relayoutData:
            # Annotation added/removed
            if 'shapes' in relayoutData:
                # globals.marked_annotations[:] = []

                if relayoutData['shapes']:
                    # For debugging
                    # print(relayoutData['shapes'][-1]['x0'])
                    # print(relayoutData['shapes'][-1]['x1'])
                    
                    check_add = []
                    for annotation in globals.marked_annotations:
                        check_add.append((annotation[0], annotation[1]))

                    check_remove = []
                    for shape in relayoutData['shapes']:
                        x0 = np.round(shape['x0'], 3)
                        x1 = np.round(shape['x1'], 3)

                        if x0 < x1:
                            annotation_start = x0
                            annotation_end = x1
                        else:
                            annotation_start = x1
                            annotation_end = x0
                        
                        check_remove.append((annotation_start, annotation_end))

                        if (annotation_start, annotation_end) not in check_add:
                            # print('new annotation {}'.format((annotation_start, annotation_end, annotation_label)))
                            globals.marked_annotations.append((annotation_start, annotation_end, annotation_label))

                    for annotation in globals.marked_annotations:
                        if (annotation[0], annotation[1]) not in check_remove:
                            # print('remove annotation {}'.format((annotation[0], annotation[1], annotation[2])))
                            globals.marked_annotations.remove((annotation[0], annotation[1], annotation[2]))

                globals.marked_annotations = merge_intervals(globals.marked_annotations)

                globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)

                quick_save(globals.raw)

            # Annotation was moved/resized
            elif json.dumps(relayoutData).startswith('shapes', 2):
                # print('Annotation was changed')
                for key in relayoutData.keys():
                    if key.endswith('x0'):
                        x0_key = key
                        # print(x0_key)
                        annotation_index = int(re.findall(r'\d+', x0_key)[0])  # returns index of changed shape
                        # print(annotation_index)
                    if key.endswith('x1'):
                        x1_key = key

                x0 = np.round(relayoutData[x0_key], 3)
                x1 = np.round(relayoutData[x1_key], 3)

                if x0 < x1:
                    annotation_start = x0
                    annotation_end = x1
                else:
                    annotation_start = x1
                    annotation_end = x0

                globals.marked_annotations[annotation_index] = (annotation_start, annotation_end, globals.marked_annotations[annotation_index][2])

                globals.marked_annotations = merge_intervals(globals.marked_annotations)

                globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)
                
                if show_annotations_only:
                    # print(annotation_index)
                    globals.current_plot_index = annotation_index

                quick_save(globals.raw)

    # Add/remove annotation label
    @app.callback(
        [Output('annotation-label', 'options'), Output('new-annotation-label', 'value'), Output('annotation-label', 'value')],
        [Input('model-output-files', 'children'), Input('new-annotation-label', 'value'), Input('remove-annotation-label', 'n_clicks')],
        [State('annotation-label', 'options'), State('annotation-label', 'value')],
        prevent_initial_call=True
    )
    def _add_annotation_label(loaded_files, new_annotation_label, remove_annotations_button, annotation_labels, current_annotation_label):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)
        print(current_annotation_label)
        print(annotation_labels)

        if 'model-output-files' in trigger:
            for file_name in loaded_files:
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

        elif 'new-annotation-label' in trigger and new_annotation_label not in globals.annotation_label_colors.keys():
            annotation_labels.append({'label': '{}'.format(new_annotation_label), 'value': '{}'.format(new_annotation_label)})
            globals.annotation_label_colors[new_annotation_label] = 'red'

        return annotation_labels, '', current_annotation_label

    # Add username to annotation labels
    @app.callback(
        [Output('annotation-label', 'options', allow_duplicate=True), Output('annotation-label', 'value', allow_duplicate=True)],
        Input('username', 'value'),
        [State('username', 'n_submit'), State('annotation-label', 'options'), State('annotation-label', 'value')],
        prevent_initial_call=True
    )
    def _add_username_to_annotation_label(username, username_changes, annotation_labels, current_annotation_label):
        # print(username_changes)
        if username:
            # print(annotation_labels)
            new_annotation_labels = annotation_labels.copy()
            
            for label_index in range(len(annotation_labels)):
                if username_changes == 1:
                    # if username was changed for the first time, add it to the end of all annotation labels
                    new_annotation_label = annotation_labels[label_index]['label'] + '_{}'.format(username)
                else:
                    # if username was changed again, replace previous one in all annotation labels
                    username_index = annotation_labels[label_index]['label'].rfind('_')
                    new_annotation_label = annotation_labels[label_index]['label'][:username_index + 1] + username

                new_annotation_labels[label_index] = {'label': '{}'.format(new_annotation_label), 'value': '{}'.format(new_annotation_label)}

            if username_changes == 1:
                new_annotation_label = current_annotation_label + '_{}'.format(username)
            else:
                username_index = current_annotation_label.rfind('_')
                new_annotation_label = current_annotation_label[:username_index + 1] + username

            # print(new_annotation_labels)
            # print(new_annotation_label)

            # Update color dict
            # print(globals.annotation_label_colors)
            annotation_label_color_keys = list(globals.annotation_label_colors.keys())
            for key in annotation_label_color_keys:
                if username_changes == 1:
                    new_key = key + '_{}'.format(username)
                else:
                    username_index = key.rfind('_')
                    new_key = key[:username_index + 1] + username
                globals.annotation_label_colors[new_key] = globals.annotation_label_colors.pop(key)
            # print(globals.annotation_label_colors)

            return new_annotation_labels, new_annotation_label

        return annotation_labels, current_annotation_label

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
            # print(annotation_label_color)
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