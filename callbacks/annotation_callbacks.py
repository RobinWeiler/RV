import json
import re

import dash
from dash.dependencies import Input, Output, State

from helperfunctions.annotation_helperfunctions import merge_intervals, annotations_to_raw
from helperfunctions.saving_helperfunctions import quick_save

import globals


def register_annotation_callbacks(app):
    # Annotation through dragging mouse across intervals callback
    @app.callback(
        Output('relayout-data', 'children'),
        Input('EEG-graph', 'relayoutData'),
        State('annotation-label', 'value'),
        prevent_initial_call=True
    )
    def _make_annotation(relayoutData, annotation_label):
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
                        x0 = round(shape['x0'], 3)
                        x1 = round(shape['x1'], 3)

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

                # print(globals.marked_annotations)

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

                x0 = round(relayoutData[x0_key], 3)
                x1 = round(relayoutData[x1_key], 3)

                if x0 < x1:
                    annotation_start = x0
                    annotation_end = x1
                else:
                    annotation_start = x1
                    annotation_end = x0

                globals.marked_annotations[annotation_index] = (annotation_start, annotation_end, globals.marked_annotations[annotation_index][2])

                globals.marked_annotations = merge_intervals(globals.marked_annotations)

                # print(globals.marked_annotations)

                globals.raw = annotations_to_raw(globals.raw, globals.marked_annotations)

                quick_save(globals.raw)

    @app.callback(
        [Output('annotation-label', 'options'), Output('new-annotation-label', 'value')],
        [Input('new-annotation-label', 'value'), Input('remove-annotation-label', 'n_clicks')],
        State('annotation-label', 'options'),
        prevent_initial_call=True
    )
    def _make_annotation(new_annotation_label, remove_annotations_button, current_annotation_labels):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'remove-annotation-label' in trigger:
            current_annotation_labels.pop()
        else:
            current_annotation_labels.append({'label': '{}'.format(new_annotation_label), 'value': '{}'.format(new_annotation_label)})

        return current_annotation_labels, ''