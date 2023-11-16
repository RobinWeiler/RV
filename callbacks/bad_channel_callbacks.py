import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

from helperfunctions.loading_helperfunctions import parse_bad_channels_file
from helperfunctions.visualization_helperfunctions import _get_list_for_displaying

import constants as c
import globals


def register_bad_channel_callbacks(app):

    # Loading bad channels and all channel names into dropdown menu and clicking channel callback
    @app.callback(
        [Output('bad-channels-dropdown', 'value'), Output('bad-channels-dropdown', 'options'), Output('selected-channels-dropdown', 'options')],
        Input('data-file', 'children'),
        [State('bad-channels-dropdown', 'value'), State('bad-channels-dropdown', 'options')],
        prevent_initial_call=True
    )
    def _update_bad_channel_dropdown(file, current_selected_bad_channels, current_available_channels):
        """Loads channel names into bad-channels-dropdown and selected-channels-dropdown. Triggers when new file is loaded and when a trace is clicked on to mark it as bad.

        Args:
            file (string): Current file-name.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.
            current_available_channels (list): List of dicts of all available channel names.

        Returns:
            tuple(list, list, list): First list contains strings of selected bad-channel names. Second and third list contain dicts of all channel names.
        """
        
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if 'data-file' in trigger and globals.raw:
            print('Loading bad channel dropdown menu...')

            all_channel_names = globals.raw.ch_names

            dropdown_channel_names = []
            for channel in all_channel_names:
                dropdown_channel_names.append({'label': channel, 'value': channel})

            bad_channels = globals.raw.info['bads']

            return bad_channels, dropdown_channel_names, dropdown_channel_names
        else:
            return [], [], []

    # Select bad-channel files
    @app.callback(
        Output('bad-channel-files', 'children'),
        Input('upload-bad-channels', 'filename'),
        prevent_initial_call=True
    )
    def _update_model_output_files(list_selected_file_names):
        """Retrieves file-names of selected bad-channel files. Triggers when new files are loaded.

        Args:
            list_selected_file_names (list): List of strings of selected bad-channel file-names.

        Returns:
            tuple(list, list): Both lists contain strings of selected model-output file-names.
        """
        if list_selected_file_names:
            print('Selected files: {}'.format(list_selected_file_names))
            return _get_list_for_displaying(list_selected_file_names)
        else:
            return []

    @app.callback(
        Output('bad-channels-dropdown', 'value', allow_duplicate=True),
        Input('upload-bad-channels', 'filename'),
        State('bad-channels-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _load_bad_channel_files(loaded_bad_channel_files, current_selected_bad_channels):
        all_loaded_bad_channels = []

        for file_name in loaded_bad_channel_files:
            if '.txt' in file_name:
                loaded_bad_channels = parse_bad_channels_file(file_name)
                all_loaded_bad_channels.append(loaded_bad_channels)

                current_selected_bad_channels += loaded_bad_channels

        current_selected_bad_channels = list(set(current_selected_bad_channels))

        disagreed_bad_channels = []
        if len(all_loaded_bad_channels) > 1:
            all_bad_channels_set = set([bad_channel for sublist in all_loaded_bad_channels for bad_channel in sublist])

            for sublist in all_loaded_bad_channels:
                sublist = set(sublist)
                bad_channels_not_in_inner_list = list(all_bad_channels_set - sublist)
                disagreed_bad_channels += bad_channels_not_in_inner_list

        globals.disagreed_bad_channels = disagreed_bad_channels

        return current_selected_bad_channels

    # Update plot when bad channels changed
    @app.callback(
        [Output('EEG-graph', 'figure', allow_duplicate=True), Output('bad-channels-dropdown', 'value', allow_duplicate=True)],
        Input('EEG-graph', 'clickData'),
        [State('EEG-graph', 'figure'), State('bad-channels-dropdown', 'value')],
        prevent_initial_call=True
    )
    def  _update_bad_channels_after_click(clickData, current_fig, current_selected_bad_channels):
        """Updates plot when bad channels changed.

        Args:
            clickData (dict): Data from latest click event.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """
        print('Clicked bad channel')

        if globals.plotting_data:
            channel_index = clickData['points'][0]['curveNumber']
            if channel_index < len(globals.plotting_data['EEG']['channel_names']):
                channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]

                patched_fig = Patch()
                
                if channel_name not in current_selected_bad_channels:
                    current_selected_bad_channels.append(channel_name)

                    patched_fig['data'][channel_index]['marker']['color'] = c.BAD_CHANNEL_COLOR

                    # If bad channels are currently hidden
                    if len(current_selected_bad_channels) > 1:
                        if not any(channel['visible'] == True and globals.plotting_data['EEG']['channel_names'][index] in current_selected_bad_channels and not globals.plotting_data['EEG']['channel_names'][index] == channel_name for index, channel in enumerate(current_fig['data'])):
                            patched_fig['data'][channel_index]['visible'] = False
                else:
                    current_selected_bad_channels.remove(channel_name)

                    patched_fig['data'][channel_index]['marker']['color'] = 'black'

                globals.raw.info['bads'] = current_selected_bad_channels

                return patched_fig, current_selected_bad_channels
        else:
            raise PreventUpdate

    # Enable/disable Hide/show bad channels button
    @app.callback(
        Output('hide-bad-channels-button', 'disabled'),
        Input('bad-channels-dropdown', 'value'),
        # prevent_initial_call=True
    )
    def _update_hide_bad_channels_button(current_selected_bad_channels):
        """Disables/enables hide-bad-channels-button. Triggered when selected bad channels change.

        Args:
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            bool: Whether or not to disable hide-bad-channels-button button.
        """

        if len(current_selected_bad_channels) > 0:
            return False
        else:
            return True

    # Hide/show bad channels
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('hide-bad-channels-button', 'n_clicks'),
        State('bad-channels-dropdown', 'value'),
        prevent_initial_call=True
    )
    def  _use_hide_bad_channels_button(hide_bad_channels, current_selected_bad_channels):
        """Hides bad channels when pressed. Shows all channels when pressed again.

        Args:
            hide_bad_channels (dict): Num clicks on hide-bad-channels-button button.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """

        if globals.plotting_data:
            patched_fig = Patch()

            for channel_name in current_selected_bad_channels:
                channel_index = globals.plotting_data['EEG']['channel_names'].index(channel_name)

                if hide_bad_channels % 2 != 0:
                    patched_fig['data'][channel_index]['visible'] = False
                else:
                    patched_fig['data'][channel_index]['visible'] = True

            return patched_fig
        else:
            raise PreventUpdate
