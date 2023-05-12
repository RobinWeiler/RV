from flask import request

from dash.dependencies import Input, Output, State

from helperfunctions.saving_helperfunctions import save_to, overwrite_save, quick_save, save_annotations, save_bad_channels

import globals


def _shutdown():
    """Shuts down server.
    """
    func = request.environ.get('werkzeug.server.shutdown')

    func()


def register_saving_callbacks(app):
    # Save button callback
    @app.callback(
        Output('save-file', 'children'),
        Input('save-button', 'n_clicks'),
        [State("save-file-name", "value"), State('extension-selection-dropdown', 'value')],
        prevent_initial_call=True
    )
    def _save_button_click(save_button, save_file_name, extension):
        """Saves data to save_file_name using given extension. Triggers when save-button is clicked. Extension defaults to .fif.

        Args:
            save_button (int): Num clicks on save-button.
            save_file_name (string): Name of save-file.
            extension (string): Save-file extension.
        """
        if not save_file_name:
            print('No file name given!')
            save_file_name = 'unnamed'
        if not extension:
            extension = '.fif'
        globals.file_name = save_to(save_file_name, extension, globals.raw)

    # Overwrite save-file button callback
    @app.callback(
        Output('overwrite-file', 'children'),
        Input('overwrite-button', 'n_clicks'),
        State('data-file', 'children'),
        prevent_initial_call=True
    )
    def _overwrite_button_click(overwrite_button, current_file_name):
        """Overwrites loaded file. This is only possible if file is located in the "save_files" directory or an external save-file-path is given.

        Args:
            overwrite_button (int): Num clicks on overwrite-button.
            current_file_name (string): File-name of plotted data.
        """
        if globals.external_save_file_path:
            overwrite_save(current_file_name, globals.raw, save_file_path=globals.external_save_file_path)
        else:
            overwrite_save(current_file_name, globals.raw)

    # Save-annotations button callback
    @app.callback(
        Output('save-annotations', 'children'),
        Input('save-annotations-button', 'n_clicks'),
        State("save-file-name", "value"),
        prevent_initial_call=True
    )
    def _save_button_click(save_annotations_button, save_file_name):
        """Saves annotations to save_file_name as .csv file. Triggers when save-annotations-button is clicked.

        Args:
            save_annotations_button (int): Num clicks on save-annotations-button.
            save_file_name (string): Name of save-file.
        """
        if not save_file_name:
            print('No file name given!')
            save_file_name = 'unnamed_annotations'
        else:
            save_file_name = save_file_name + '_annotations'

        save_annotations(save_file_name, globals.raw)

    # Save-bad-channels button callback
    @app.callback(
        Output('save-bad-channels', 'children'),
        Input('save-bad-channels-button', 'n_clicks'),
        State("save-file-name", "value"),
        prevent_initial_call=True
    )
    def _save_button_click(save_bad_channels_button, save_file_name):
        """Saves annotations to save_file_name as .csv file. Triggers when save-annotations-button is clicked.

        Args:
            save_bad_channels_button (int): Num clicks on save-bad-channels-button.
            save_file_name (string): Name of save-file.
        """
        if not save_file_name:
            print('No file name given!')
            save_file_name = 'unnamed_bad_channels'
        else:
            save_file_name = save_file_name + '_bad_channels'

        save_bad_channels(save_file_name, globals.raw)

    # Quit button callback
    @app.callback(
        Output('quit-viewer', 'children'),
        Input('final-quit-button', 'n_clicks'),
        State('data-file', 'children'),
        prevent_initial_call=True
    )
    def quit_button_click(quit_button, current_file_name):
        """Shuts down server. If external save-file-path is given, data is saved by overwriting given path.

        Args:
            quit_button (int): Num clicks on quit-button.
            current_file_name (string): File-name of plotted data.
        """
        quick_save(globals.raw)

        if globals.external_save_file_path:
            overwrite_save(current_file_name, globals.raw, save_file_path=globals.external_save_file_path)

        print('Shutting down server')

        _shutdown()

    app.clientside_callback(
        '''
            function close(n){
                        // sleep time expects milliseconds
                function sleep (time) {
                  return new Promise((resolve) => setTimeout(resolve, time));
                }
                if(n){
                    //leave some time for the other callback
                    sleep(1000).then(() => {
                        window.close()
                    });
                }
            }
        ''',
        Output('quit-viewer-close', 'children'),
        Input('final-quit-button', 'n_clicks'),
        prevent_initial_call=True
    )
