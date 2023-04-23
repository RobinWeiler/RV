import dash
from dash.dependencies import Input, Output


def register_model_callbacks(app):
    # Model output selection callback
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
        button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'reset-models' in button_pressed:
            print('Resetting loaded model-predictions')
            return [], None
        elif list_selected_file_names:
            print('Selected files: {}'.format(list_selected_file_names))
            return list_selected_file_names, list_selected_file_names

    # Model threshold disable callback
    @app.callback(
        Output('model-threshold', 'disabled'),
        Input('annotate-model', 'value'),
        # prevent_initial_call=True
    )
    def _disable_model_threshold(model_annotate):
        if model_annotate:
            return False
        else:
            return True