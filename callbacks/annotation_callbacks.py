import json
import re

import dash
from dash import dcc, html, Input, Output, State, Patch, ALL
from dash.exceptions import PreventUpdate

import numpy as np

from helperfunctions.annotation_helperfunctions import merge_intervals, annotations_to_raw, get_annotations, _get_annotation_label_radioitem
from helperfunctions.loading_helperfunctions import parse_annotation_file
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.saving_helperfunctions import quick_save

import globals
import constants as c


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
        Output('hidden-annotation-output', 'n_clicks', allow_duplicate=True),
        Input('EEG-graph', 'relayoutData'),
        [State('show-annotations-only', 'value'), State('EEG-graph', 'figure'), State('username', 'value')],
        prevent_initial_call=True
    )
    def _make_annotation(relayoutData, show_annotations_only, current_fig, username):
        """Saves annotations when new ones are made or old ones are moved/deleted. Triggers when user zooms, pans, and draws on plot.

        Args:
            relayoutData (dict): Data from latest relayout event.
            annotation_label (string); Label for new annotations.
        """
        # print(relayoutData)
        if relayoutData:
            # Annotation was added/removed/moved/resized
            if any('shapes' in key for key in relayoutData.keys()):
                globals.plotting_data['annotations']['marked_annotations'][:] = []
                redraw_needed = False

                for shape in current_fig['layout']['shapes']:
                    # print(shape)
                    if shape['type'] == 'rect':
                        x0 = np.round(shape['x0'], 3)
                        x1 = np.round(shape['x1'], 3)
                        # if 'label' in shape.keys() and shape['label']['text']:
                        #     label = shape['label']['text']
                        # else:
                        label = shape['name']

                        if x0 < x1:
                            annotation_start = x0
                            annotation_end = x1
                        else:
                            annotation_start = x1
                            annotation_end = x0

                        if annotation_start < 0:
                            annotation_start = 0
                            redraw_needed = True
                        if annotation_end > globals.plotting_data['EEG']['recording_length']:
                            annotation_end = globals.plotting_data['EEG']['recording_length']
                            redraw_needed = True

                        globals.plotting_data['annotations']['marked_annotations'].append((annotation_start, annotation_end, label))
                        # current_annotations.append((annotation_start, annotation_end))

                globals.plotting_data['annotations']['marked_annotations'], merge_happened = merge_intervals(globals.plotting_data['annotations']['marked_annotations'])
                print(globals.plotting_data['annotations']['marked_annotations'])

                globals.raw = annotations_to_raw(globals.raw, globals.plotting_data['annotations']['marked_annotations'], username)
                quick_save(globals.raw)

                if show_annotations_only or redraw_needed or merge_happened:
                    return 1

        raise PreventUpdate

    # Update segment-slider when marked annotations changed and view-annotations-only mode is used
    @app.callback(
        [
            # Output('EEG-graph', 'figure', allow_duplicate=True),
            Output('segment-slider', 'max', allow_duplicate=True), Output('segment-slider', 'step', allow_duplicate=True),
            Output('segment-slider', 'marks', allow_duplicate=True), Output('segment-slider', 'value', allow_duplicate=True),
        ],
        Input('hidden-annotation-output', 'n_clicks'),
        [State('show-annotations-only', 'value'), State('segment-slider', 'value')],
        prevent_initial_call=True
    )
    def _update_segment_slider_annotations_only_mode(hidden_output, show_annotations_only, current_segment):
        """Updates segment-slider when new annotations are made or old ones are moved/deleted.

        Args:
            
        """
        if show_annotations_only:
            if len(globals.plotting_data['annotations']['marked_annotations']) == 0:
                raise Exception('There are no artifacts to show')
            else:
                num_segments = int(len(globals.plotting_data['annotations']['marked_annotations']) - 1)
                marks = {i: '{}'.format(i) for i in range(num_segments + 1)}

                if current_segment >= num_segments:
                    current_segment = num_segments

                return num_segments, 1, marks, current_segment
        else:
            raise PreventUpdate

    # Add/remove/rename annotation label
    @app.callback(
        [Output('annotation-label', 'options'), Output('new-annotation-label', 'value'), Output('annotation-label', 'value')],
        [Input('data-file', 'children'), Input('upload-model-output', 'filename'), Input('new-annotation-label', 'value'), Input('remove-annotation-label', 'n_clicks'), Input('rename-annotation-label', 'n_clicks'), Input("annotate-model", "value")],
        [State('annotation-label', 'options'), State('annotation-label', 'value'), State('renamed-annotation-label', 'value'), State('username', 'value')],
        prevent_initial_call=True
    )
    def _add_annotation_label(current_file_name, loaded_annotation_files, new_annotation_label, remove_annotations_button, rename_annotation_label_button, model_annotate, annotation_labels, current_annotation_label, renamed_annotation_label, username):
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)

        if 'data-file' in trigger:
            if globals.raw:
                loaded_annotations = get_annotations(globals.raw)

                if 'annotations' in globals.parameters.keys():
                    loaded_annotations += globals.parameters['annotations']

                for annotation in loaded_annotations:
                    if annotation[2] not in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                        globals.plotting_data['annotations']['annotation_label_colors'][annotation[2]] = 'red'
                        annotation_labels.append(_get_annotation_label_radioitem(annotation[2]))

        elif 'upload-model-output' in trigger:
            if loaded_annotation_files:
                for file_name in loaded_annotation_files:
                    if '.csv' in file_name:
                        loaded_annotations = parse_annotation_file(file_name)

                        for annotation in loaded_annotations:
                            if annotation[2] not in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                                globals.plotting_data['annotations']['annotation_label_colors'][annotation[2]] = 'red'
                                annotation_labels.append(_get_annotation_label_radioitem(annotation[2]))
                    else:
                        if globals.plotting_data['annotations']['default_model_annotation_label'] not in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                            annotation_labels.append(_get_annotation_label_radioitem(globals.plotting_data['annotations']['default_model_annotation_label']))
                            globals.plotting_data['annotations']['annotation_label_colors'][globals.plotting_data['annotations']['default_model_annotation_label']] = 'red'

        elif 'remove-annotation-label' in trigger:
            globals.plotting_data['annotations']['marked_annotations'] = [annotation for annotation in globals.plotting_data['annotations']['marked_annotations'] if annotation[2] != current_annotation_label]
            globals.raw = annotations_to_raw(globals.raw, globals.plotting_data['annotations']['marked_annotations'], username)
            quick_save(globals.raw)

            if len(annotation_labels) > 1:
                for annotation_label in annotation_labels:
                    if annotation_label['value'] == current_annotation_label:
                        annotation_labels.remove(annotation_label)

                globals.plotting_data['annotations']['annotation_label_colors'].pop(current_annotation_label)
                
                current_annotation_label = annotation_labels[0]['value']

        elif 'rename-annotation-label' in trigger and len(annotation_labels) > 0:
            for annotation_index, annotation_label in enumerate(annotation_labels):
                if annotation_label['value'] == current_annotation_label:
                    annotation_labels[annotation_index] = _get_annotation_label_radioitem(renamed_annotation_label)
                    break

            globals.plotting_data['annotations']['annotation_label_colors'][renamed_annotation_label] = globals.plotting_data['annotations']['annotation_label_colors'].pop(current_annotation_label)
            if current_annotation_label == globals.plotting_data['annotations']['default_model_annotation_label']:
                globals.plotting_data['annotations']['default_model_annotation_label'] = renamed_annotation_label

            globals.plotting_data['annotations']['marked_annotations'] = [(annotation[0], annotation[1], renamed_annotation_label) if annotation[2] == current_annotation_label else annotation for annotation in globals.plotting_data['annotations']['marked_annotations']]
            globals.raw = annotations_to_raw(globals.raw, globals.plotting_data['annotations']['marked_annotations'], username)
            quick_save(globals.raw)

            current_annotation_label = renamed_annotation_label

        elif 'new-annotation-label' in trigger:
            if new_annotation_label not in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                annotation_labels.append(_get_annotation_label_radioitem(new_annotation_label))
            else:
                raise PreventUpdate

        elif 'annotate-model' in trigger:
            if model_annotate and globals.plotting_data['annotations']['default_model_annotation_label'] not in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                annotation_labels.append(_get_annotation_label_radioitem(globals.plotting_data['annotations']['default_model_annotation_label']))
            else:
                raise PreventUpdate

        return annotation_labels, '', current_annotation_label

    # Change username
    @app.callback(
        Output('username-dummy', 'children', allow_duplicate=True),
        Input('username', 'value'),
        prevent_initial_call=True
    )
    def _change_username(username):
        print(username)
        if not globals.raw:
            raise PreventUpdate
        else:
            globals.raw = annotations_to_raw(globals.raw, globals.plotting_data['annotations']['marked_annotations'], username)
            quick_save(globals.raw)

    # Annotation-label color has changed
    @app.callback(
        Output('hidden-annotation-output', 'n_clicks'),
        [Input({'type': 'color-dropdown', 'label': ALL}, 'value'), Input('annotation-label', 'options')],
    )
    def _choose_annotation_label_color(annotation_label_colors, annotation_labels):
        trigger = dash.callback_context.triggered_id
        print(trigger)

        for label_index in range(len(annotation_labels)):
            globals.plotting_data['annotations']['annotation_label_colors'][annotation_labels[label_index]['value']] = annotation_label_colors[label_index]

        return 1

    # Update plot when selected annotation-label is changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('annotation-label', 'value'),
        State('show-annotation-labels', 'value'),
        prevent_initial_call=True
    )
    def _update_new_annotations(current_annotation_label, show_annotation_labels):
        """Updates annotations when annotation label or color are changed.

        Args:
            current_annotation_label (string); Label for new annotations.

        Returns:
            tuple(dash.Patch(), int): Updated EEG plot.
        """
        if globals.plotting_data['EEG']:
            print(current_annotation_label)

            patched_fig = Patch()

            if globals.plotting_data['annotations']['annotation_label_colors'][current_annotation_label] != 'hide':
                patched_fig['layout']['newshape']['fillcolor'] = globals.plotting_data['annotations']['annotation_label_colors'][current_annotation_label]
            else:
                patched_fig['layout']['newshape']['visible'] = False

            patched_fig['layout']['newshape']['name'] = current_annotation_label
            # patched_fig['layout']['newshape']['legendgroup'] = annotation_label
            
            if show_annotation_labels:
                patched_fig['layout']['newshape']['label']['text'] = current_annotation_label
            
            return patched_fig
        raise PreventUpdate

    # Update plot when annotation-label or annotation-label-color is changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        [Input('hidden-annotation-output', 'n_clicks'), Input('show-annotation-labels', 'value')],
        [State('EEG-graph', 'figure'), State('annotation-label', 'value')],
        prevent_initial_call=True
    )
    def _update_annotations(hidden_annotation_output, show_annotation_labels, current_fig, current_annotation_label):
        """Updates annotations when annotation label or color are changed.

        Args:
            annotation_label (string); Label for new annotations.
            annotation_label_color (dict); Color for new annotations.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(plotly.graph_objs.Figure, int): Updated EEG plot.
        """
        if globals.plotting_data['EEG']:
            trigger = dash.callback_context.triggered_id
            print(trigger)

            patched_fig = Patch()

            if globals.plotting_data['annotations']['annotation_label_colors'][current_annotation_label] != 'hide':
                patched_fig['layout']['newshape']['fillcolor'] = globals.plotting_data['annotations']['annotation_label_colors'][current_annotation_label]
            else:
                patched_fig['layout']['newshape']['visible'] = False

            if 'show-annotation-labels' in trigger:
                if show_annotation_labels:
                    new_yaxis_end = current_fig['layout']['yaxis']['range'][1] + (4 * c.DEFAULT_Y_AXIS_OFFSET)
                    patched_fig['layout']['newshape']['label']['text'] = current_annotation_label
                else:
                    new_yaxis_end = current_fig['layout']['yaxis']['range'][1] - (4 * c.DEFAULT_Y_AXIS_OFFSET)
                    patched_fig['layout']['newshape']['label']['text'] = ''

                patched_fig['layout']['yaxis']['range'][1] = new_yaxis_end
                patched_fig['layout']['updatemenus'][0]['buttons'][1]['args'][0]['yaxis.range[1]'] = new_yaxis_end  # this only takes effect after switching segments

            patched_fig['layout']['shapes'] = []
            for annotation in globals.plotting_data['annotations']['marked_annotations']:
                patched_fig['layout']['shapes'].append({
                    'editable': True,
                    'xref': 'x',
                    'yref': 'y',
                    'layer': 'below',
                    'opacity': 0.6,
                    'line': {'width': 0},
                    'fillcolor': globals.plotting_data['annotations']['annotation_label_colors'][annotation[2]] if globals.plotting_data['annotations']['annotation_label_colors'][annotation[2]] != 'hide' else 'red',
                    'fillrule': 'evenodd',
                    'type': 'rect',
                    'x0': annotation[0],
                    'y0': current_fig['layout']['yaxis']['range'][0],  # len(globals.plotting_data['EEG']['channel_names']) * globals.plotting_data['plot']['offset_factor'] + globals.plotting_data['plot']['offset_factor'],
                    'x1': annotation[1],
                    'y1': current_fig['layout']['yaxis']['range'][1] if not 'show-annotation-labels' in trigger else new_yaxis_end,
                    'name': annotation[2],
                    'label':{'text': annotation[2] if show_annotation_labels else '', 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}},
                    'visible': True if globals.plotting_data['annotations']['annotation_label_colors'][annotation[2]] != 'hide' else False
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
