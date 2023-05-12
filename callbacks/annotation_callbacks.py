import json
import re

import dash
from dash.dependencies import Input, Output, State

import numpy as np

from helperfunctions.annotation_helperfunctions import merge_intervals, annotations_to_raw
from helperfunctions.loading_helperfunctions import parse_annotation_file
from helperfunctions.saving_helperfunctions import quick_save

import globals


def register_annotation_callbacks(app):
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
                    print(annotation_index)
                    globals.current_plot_index = annotation_index

                quick_save(globals.raw)

    @app.callback(
        [Output('annotation-label', 'options'), Output('new-annotation-label', 'value'), Output('annotation-label', 'value')],
        [Input('model-output-files', 'children'), Input('new-annotation-label', 'value'), Input('remove-annotation-label', 'n_clicks')],
        [State('annotation-label', 'options'), State('annotation-label', 'value')],
        prevent_initial_call=True
    )
    # Adds new annotation label callback
    def _add_annotation_label(loaded_files, new_annotation_label, remove_annotations_button, annotation_labels, current_annotation_label):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)
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
            if current_annotation_label == annotation_labels[-1]['value']:
                current_annotation_label = annotation_labels[0]['value']
            annotation_labels.pop()

        elif 'new-annotation-label' in trigger and new_annotation_label not in globals.annotation_label_colors.keys():
            annotation_labels.append({'label': '{}'.format(new_annotation_label), 'value': '{}'.format(new_annotation_label)})
            globals.annotation_label_colors[new_annotation_label] = 'red'

        return annotation_labels, '', current_annotation_label

    @app.callback(
        Output('annotation-label-color', 'value'),
        Input('annotation-label', 'value'),
        # State('annotation-label', 'options'),
        prevent_initial_call=True
    )
    # Switch to current annotation-label color callback
    def _switch_annotation_label_color(current_annotation_label):
        color = globals.annotation_label_colors[current_annotation_label]
        
        return color

    @app.callback(
        Output('chosen-annotation-color', 'children'),
        Input('annotation-label-color', 'value'),
        State('annotation-label', 'value'),
        prevent_initial_call=True
    )
    # Choose annotation-label color callback
    def _choose_annotation_label_color(current_annotation_label_color, current_annotation_label):
        globals.annotation_label_colors[current_annotation_label] = current_annotation_label_color
