from dash.dependencies import Input, Output

from helperfunctions.loading_helperfunctions import parse_data_file

import globals


def register_loading_callbacks(app):
    # Data-file selection callback
    @app.callback(
        [Output('data-file', 'children'), Output("high-pass", "value"), Output("low-pass", "value"), Output("resample-rate", "value"), Output('upload-file', 'disabled')],
        Input('upload-file', 'filename'),
    )
    def _load_file(selected_file_name):
        """Sets file-name, high-pass filter, low-pass filter, and sampling frequency based on loaded data. If external Raw with specified parameters is given they are used instead. Using external Raw disables file selection. Triggers when file is selected.

        Args:
            selected_file_name (string): File-name of selected recording.

        Returns:
            (string, float, float, float, bool): File-name, high-pass filter, low-pass filter, sampling frequency, and whether or not to disable file selection.
        """
        if globals.external_raw:
            file_name_index = globals.raw._filenames[0].rfind('/')   # external_save_file_path.rfind('/')
            external_file_name = globals.raw._filenames[0][file_name_index + 1:]  # external_save_file_path[file_name_index + 1:]
            globals.file_name = external_file_name

            loaded_highpass = globals.raw.info['highpass']
            loaded_lowpass = globals.raw.info['lowpass']
            loaded_sfreq = globals.raw.info['sfreq']
            
            # overwrite with loaded parameter
            if 'highpass' in globals.parameters.keys():
                loaded_highpass = globals.parameters['highpass']
            if 'lowpass' in globals.parameters.keys():
                loaded_lowpass = globals.parameters['lowpass']
            if 'sfreq' in globals.parameters.keys():
                loaded_sfreq = globals.parameters['sfreq']

            return external_file_name, loaded_highpass, loaded_lowpass, loaded_sfreq, True

        elif selected_file_name:
            globals.file_name = str(selected_file_name)
            # print(selected_file_name)

            globals.raw = parse_data_file(selected_file_name)

            loaded_highpass = globals.raw.info['highpass']
            loaded_lowpass = globals.raw.info['lowpass']
            loaded_sfreq = globals.raw.info['sfreq']
            
            # overwrite with loaded parameter
            if 'highpass' in globals.parameters.keys():
                loaded_highpass = globals.parameters['highpass']
            if 'lowpass' in globals.parameters.keys():
                loaded_lowpass = globals.parameters['lowpass']
            if 'sfreq' in globals.parameters.keys():
                loaded_sfreq = globals.parameters['sfreq']

            return selected_file_name, loaded_highpass, loaded_lowpass, loaded_sfreq, False

        else:
            return None, None, None, None, False
