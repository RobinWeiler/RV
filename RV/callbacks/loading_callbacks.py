import os
import shutil

from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback
from dash.exceptions import PreventUpdate

from RV.callbacks.utils.annotation_utils import get_annotation_label_radioitem
from RV.callbacks.utils.channel_selection_utils import get_channel_topography_plot
from RV.callbacks.utils.loading_utils import load_raw


def register_loading_callbacks(auto_save=False, external_data=False):
    # Get width of RV-main-graph in pixels to use as default resample-points
    clientside_callback(
        """
            function(id) {
                const mainGraph = document.getElementById('RV-main-graph');
                const resamplePoints = mainGraph.offsetWidth;
                return resamplePoints
            }
        """,
        Output('RV-resample-points-input', 'value'),
        Input('RV-resample-points-input', 'id'),
    )

    @callback(
        [
            Output('RV-file-paths', 'data', allow_duplicate=True),
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-plotting-data', 'data', allow_duplicate=True),
            Output('RV-model-data', 'data', allow_duplicate=True),
            Output('RV-clear-main-graph-button', 'n_clicks', allow_duplicate=True),
            Output('RV-high-pass-input', 'value', allow_duplicate=True), Output('RV-low-pass-input', 'value', allow_duplicate=True),
            Output('RV-channel-selection-dropdown', 'options', allow_duplicate=True),
            Output('RV-channel-selection-graph', 'figure', allow_duplicate=True), Output('RV-channel-selection-graph', 'style', allow_duplicate=True),
            Output('RV-bad-channels-dropdown', 'options', allow_duplicate=True), Output('RV-bad-channels-dropdown', 'value', allow_duplicate=True),
            Output('RV-bad-channel-file-selection-dropdown', 'value', allow_duplicate=True),
            Output('RV-annotation-file-selection-dropdown', 'value', allow_duplicate=True),
            Output('RV-annotation-label', 'options', allow_duplicate=True),
            Output('RV-segment-slider', 'max', allow_duplicate=True), Output('RV-segment-slider', 'marks', allow_duplicate=True), Output('RV-segment-slider', 'value', allow_duplicate=True),
            Output({'type': 'open-button', 'modal': 'RV-stats'}, 'disabled', allow_duplicate=True), Output('RV-open-stats-button-2', 'disabled', allow_duplicate=True)
        ],
        Input('RV-file-selection-dropdown', 'value'),
        [
            State('RV-file-paths', 'data'),
            State('RV-segment-size-input', 'value'),
            State('RV-annotation-label', 'options'),
            State('RV-annotations-only-mode', 'value')
        ],
        # prevent_initial_call=True
    )
    def load_file(selected_file, file_paths, segment_size, annotation_label_options, annotations_only_mode):
        """Loads mne.io.Raw object from selected_file. Loads various attributes of loaded data into RV-settings modal. 
        Also clears Serverside cache.
        """
        if selected_file:
            print(selected_file)

            # Clear Serverside cache
            serverside_cache = file_paths['serverside_cache']
            if os.path.exists(serverside_cache):
                for filename in os.listdir(serverside_cache):
                    if filename == 'info.txt':
                        continue
                    else:
                        file_path = os.path.join(serverside_cache, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            if auto_save and not external_data:
                save_file_name = os.path.basename(selected_file)
                base_name, extension = os.path.splitext(save_file_name)

                if extension != '.fif':
                    save_file_name += '.fif'

                # Switch target of quick_save callback (saving_callbacks.py) to auto-save to file_paths['save_file_path']
                file_paths['temp_save_file_path'] = os.path.join(file_paths['save_file_path'], save_file_name)

            raw = load_raw(selected_file)
            print(raw.info)
            print(raw.annotations)

            raw = raw.pick('data')

            plotting_data = {'selected_channels': raw.ch_names, 'recording_length': (len(raw) / raw.info['sfreq'])}
            plotting_data['bad_channels'] = {str(selected_file): raw.info['bads']}

            channel_selection_options = [{'label': channel, 'value': channel} for channel in raw.ch_names]
            if raw.info['dig']:
                channel_topography_plot = get_channel_topography_plot(raw)
                channel_topography_plot_style = {}
            else:
                channel_topography_plot = None
                channel_topography_plot_style = {'display': 'none'}

            loaded_annotation_labels = list(set(raw.annotations.description))  # without duplicates
            current_annotation_labels = [annotation_label['value'] for annotation_label in annotation_label_options]
            for annotation_label in loaded_annotation_labels:
                if annotation_label not in current_annotation_labels:
                    annotation_label_options.append(get_annotation_label_radioitem(annotation_label)[0])

            # RV-segment-slider parameters
            if annotations_only_mode:
                num_segments = len(raw.annotations) - 1
                segment_ticks = {i: {'label': f'{i}'} for i in range(len(raw.annotations))}
            elif segment_size:
                num_segments = int(plotting_data['recording_length'] // segment_size)
                segment_ticks = {i: {'label': f'{i * segment_size}'} for i in range(num_segments + 1)}
            else:
                num_segments = 0
                segment_ticks = {0: {'label': '0'}}

            return file_paths, \
                Serverside(raw), \
                plotting_data, \
                {}, \
                1, \
                raw.info['highpass'], raw.info['lowpass'], \
                channel_selection_options, \
                channel_topography_plot, channel_topography_plot_style, \
                channel_selection_options, raw.info['bads'], \
                None, \
                None, \
                annotation_label_options, \
                num_segments, segment_ticks, None, \
                False, False

        raise PreventUpdate
