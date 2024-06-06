import os

from dash_extensions.enrich import Output, Input, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

from RV.callbacks.utils.loading_utils import get_file_selection_options
from RV.callbacks.utils.saving_utils import save_annotations_csv, save_bad_channels_txt


def register_saving_callbacks():
    @callback(
        Output({'type': 'modal', 'modal': 'RV-save'}, 'is_open', allow_duplicate=True),
        Input({'type': 'modal', 'modal': 'RV-overwrite'}, 'is_open'),
        prevent_initial_call=True
    )
    def toggle_save_modal(overwrite_is_open):
        """Toggles RV-save modal opposite to RV-overwrite modal.
        """
        return not overwrite_is_open

    @callback(
        Output('RV-raw', 'data'),
        Input('RV-raw', 'data'),
        State('RV-file-paths', 'data'),
        prevent_initial_call=True
    )
    def quick_save(raw, file_paths):
        """Saves mne.io.Raw object to file_paths['temp_save_file_path']. Triggered by changes to Raw object.
        """
        if raw is None:
            raise PreventUpdate

        raw.save(file_paths['temp_save_file_path'], overwrite=True)

        raise PreventUpdate

    @callback(
        [
            Output({'type': 'open-button', 'modal': 'RV-overwrite'}, 'n_clicks', allow_duplicate=True),
            Output('RV-overwrite-save-file', 'children', allow_duplicate=True),
            Output('RV-file-selection-dropdown', 'options', allow_duplicate=True),
            Output('RV-bad-channel-file-selection-dropdown', 'options', allow_duplicate=True),
            Output('RV-annotation-file-selection-dropdown', 'options', allow_duplicate=True)
        ],
        [
            Input('RV-save-button', 'n_clicks'),
            Input('RV-save-annotations-button', 'n_clicks'),
            Input('RV-save-bad-channels-button', 'n_clicks'),
        ],
        [
            State('RV-raw', 'data'),
            State('RV-file-paths', 'data'),
            State('RV-save-file-name', 'value')
        ],
        prevent_initial_call=True
    )
    def save(save, save_annotations, save_bad_channels, raw, file_paths, save_file_name):
        """Saves mne.io.Raw object to .fif file if RV-save-button is clicked.
        Saves annotations to .csv file if RV-save-annotations-button is clicked.
        Saves bad channels to .txt file if RV-save-bad-channels-button is clicked.

        Files are saved with entered save-file name to file_paths['save_file_path'].
        Opens RV-overwrite modal if save_file_name already exists.
        Also refreshes file-selection-dropdown options (also for bad-channel files and annotation files).
        Triggered by RV-save-button, RV-save-annotations-button, and RV-save-bad-channels-button.
        """
        if not save_file_name:
            raise Exception('No save-file name was given.')

        trigger = ctx.triggered_id
        # print(trigger)

        base_name, extension = os.path.splitext(save_file_name)

        # Add respective extension if necessary
        if trigger == 'RV-save-button':
            if extension != '.fif':
                save_file_name += '.fif'
        elif trigger == 'RV-save-annotations-button':
            if extension != '.csv':
                save_file_name += '.csv'
        elif trigger == 'RV-save-bad-channels-button':
            if extension != '.txt':
                save_file_name += '.txt'
        else:
            raise Exception(f'Unknown trigger encountered: {trigger}')

        save_file_path = os.path.join(file_paths['save_file_path'], save_file_name)

        open_overwrite = no_update
        if os.path.exists(save_file_path):
            print(f"{save_file_name} already exists at {file_paths['save_file_path']}")

            # Open RV-overwrite modal
            open_overwrite = 1
        elif trigger == 'RV-save-button':
            raw.save(save_file_path)
        elif trigger == 'RV-save-annotations-button':
            save_annotations_csv(raw, save_file_path)
        elif trigger == 'RV-save-bad-channels-button':
            save_bad_channels_txt(raw, save_file_path)

        return open_overwrite, save_file_path, \
            get_file_selection_options(file_paths, ['.fif', '.raw', '.edf', '.bdf', '.set']), \
            get_file_selection_options(file_paths, ['.txt']), \
            get_file_selection_options(file_paths, ['.csv', '.fif'])

    @callback(
        [
            Output('RV-file-selection-dropdown', 'options', allow_duplicate=True),
            Output('RV-bad-channel-file-selection-dropdown', 'options', allow_duplicate=True),
            Output('RV-annotation-file-selection-dropdown', 'options', allow_duplicate=True)
        ],
        Input({'type': 'process-button', 'modal': 'RV-overwrite'}, 'n_clicks'),
        [
            State('RV-raw', 'data'),
            State('RV-file-paths', 'data'),
            State('RV-overwrite-save-file', 'children')
        ],
        prevent_initial_call=True
    )
    def overwrite_save_file(overwrite_button, raw, file_paths, save_file_path):
        """Overwrites save_file_path with current mne.io.Raw object.
        Also refreshes file-selection-dropdown options (also for bad-channel files and annotation files).
        Triggered by RV-overwrite button.
        """
        if overwrite_button:
            base_name, extension = os.path.splitext(save_file_path)

            if extension == '.fif':
                raw.save(save_file_path, overwrite=True)
            elif extension == '.csv':
                save_annotations_csv(raw, save_file_path)
            elif extension == '.txt':
                save_bad_channels_txt(raw, save_file_path)
            else:
                raise NotImplementedError(f'Unknown file extension: {extension}.')

            return get_file_selection_options(file_paths, ['.fif', '.raw', '.edf', '.bdf', '.set']), \
                get_file_selection_options(file_paths, ['.txt']), \
                get_file_selection_options(file_paths, ['.csv', '.fif'])

        raise PreventUpdate
