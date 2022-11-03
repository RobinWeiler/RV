from dash.dependencies import Input, Output, State
import dash

from helperfunctions.loading_helperfunctions import parse_data_file

import globals
import constants as c


def register_loading_callbacks(app):
    # Data-file selection callback
    @app.callback(
        [
            Output('data-file', 'children'), 
            Output("high-pass", "value"), Output("low-pass", "value"), Output('reference-dropdown', 'value'),
            Output("resample-rate", "value"), Output('scale', 'value'), Output('channel-offset', 'value'), Output('segment-size', 'value'),
            Output('upload-file', 'disabled')
        ],
        [
            Input('upload-file', 'filename'),
            Input("low-pass", "value")
        ],
        [
            State('data-file', 'children'), State("high-pass", "value"), State('reference-dropdown', 'value'),
            State("scale", "value"), State("channel-offset", "value"), State('segment-size', 'value'),
            State('upload-file', 'disabled')
        ]
    )
    def _load_file(selected_file_name, low_pass, current_file_name, high_pass, reference,
                        scale, channel_offset, segment_size,
                        disable_upload):
        """Sets file-name, highpass filter, lowpass filter, and sampling frequency (3 times lowpass-filter parameter) based on loaded data. If external Raw with specified parameters is given they are used instead. Using external Raw disables file selection. Triggers when file is selected.

        Args:
            selected_file_name (string): File-name of selected recording.
            low_pass (float): Input desired low-pass filter value.
            current_file_name (string): File-name of currently loaded EEG recording.
            high_pass (float): Input desired high-pass filter value.
            reference (string): Chosen reference.
            scale (float): Input desired scaling for data.
            channel_offset (float): Input desired channel offset.
            segment_size (int): Input desired segment size for plots.
            disable_upload (bool): Whether or not file selection is disabled.

        Returns:
            (string, float, float, float, bool): File-name, highpass filter, lowpass filter, sampling frequency, and whether or not to disable file selection.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'low-pass.value' in trigger:
            # Adjust resampling frequency if lowpass-filter has changed
            new_sampling_frequency = 3 * low_pass
            return current_file_name, high_pass, low_pass, reference, new_sampling_frequency, scale, channel_offset, segment_size, disable_upload
        elif globals.external_raw or selected_file_name:
            if globals.external_raw:
                file_name_index = globals.raw._filenames[0].rfind('/')   # external_save_file_path.rfind('/')
                external_file_name = globals.raw._filenames[0][file_name_index + 1:]  # external_save_file_path[file_name_index + 1:]
                globals.file_name = external_file_name
                selected_file_name = external_file_name
                
                file_selection_disabled = True
            else:
                globals.file_name = str(selected_file_name)
                
                globals.raw = parse_data_file(selected_file_name)
                
                file_selection_disabled = False

            loaded_highpass = globals.raw.info['highpass']
            loaded_lowpass = globals.raw.info['lowpass']
            loaded_sfreq = 3 * globals.raw.info['lowpass']
            loaded_reference = 'None'
            loaded_scale = None
            loaded_offset = None
            loaded_segment_size = c.DEFAULT_SEGMENT_SIZE
            
            # overwrite with loaded parameter
            if 'high_pass' in globals.parameters.keys():
                loaded_highpass = globals.parameters['high_pass']
            if 'low_pass' in globals.parameters.keys():
                loaded_lowpass = globals.parameters['low_pass']
            if 'reference' in globals.parameters.keys():
                loaded_reference = globals.parameters['reference']
            if 'resampling_rate' in globals.parameters.keys():
                loaded_sfreq = globals.parameters['resampling_rate']
            if 'scale' in globals.parameters.keys():
                loaded_scale = globals.parameters['scale']
            if 'offset' in globals.parameters.keys():
                loaded_offset = globals.parameters['offset']
            if 'segment_size' in globals.parameters.keys():
                loaded_segment_size = globals.parameters['segment_size']

            return selected_file_name, loaded_highpass, loaded_lowpass, loaded_reference, loaded_sfreq, loaded_scale, loaded_offset, loaded_segment_size, file_selection_disabled
        else:
            return None, None, None, None, None, None, None, None, False
