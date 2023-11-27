import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

from helperfunctions.loading_helperfunctions import parse_bad_channels_file
from helperfunctions.visualization_helperfunctions import _get_list_for_displaying

import constants as c
import globals


def register_bad_channel_callbacks(app):

    # Loading bad channels and all channel names into dropdown menu callback
    @app.callback(
        [Output('bad-channels-dropdown', 'value'), Output('bad-channels-dropdown', 'options'), Output('selected-channels-dropdown', 'options'), Output('bad-channel-files', 'children', allow_duplicate=True)],
        Input('data-file', 'children'),
        prevent_initial_call=True
    )
    def _update_bad_channel_dropdown(file):
        """Loads channel names into bad-channels-dropdown and selected-channels-dropdown. Triggers when new file is loaded .

        Args:
            file (string): Current file-name.

        Returns:
            tuple(list, list, list): First list contains strings of selected bad-channel names. Second and third list contain dicts of all channel names.
        """
        globals.bad_channels = {}

        if globals.raw:
            print('Loading bad channel dropdown menu...')

            all_channel_names = globals.raw.ch_names

            dropdown_channel_names = []
            for channel in all_channel_names:
                dropdown_channel_names.append({'label': channel, 'value': channel})

            loaded_bad_channels = globals.raw.info['bads']
            globals.bad_channels[file] = loaded_bad_channels

            if 'bad_channels' in globals.parameters.keys():
                all_loaded_bad_channels = globals.parameters['bad_channels']
                
                for annotator, bad_channels in all_loaded_bad_channels.items():
                    globals.bad_channels[annotator] = bad_channels
                    loaded_bad_channels += bad_channels

            loaded_bad_channels = list(set(loaded_bad_channels))

            return loaded_bad_channels, dropdown_channel_names, dropdown_channel_names, []
        else:
            return [], [], [], []

    # Select bad-channel files callback
    @app.callback(
        [Output('bad-channel-files', 'children'), Output('upload-bad-channels', 'filename')],
        [Input('upload-bad-channels', 'filename'), Input('reset-bad-channels', 'n_clicks')],
        prevent_initial_call=True
    )
    def _update_bad_channel_files(list_selected_file_names, reset_bad_channels):
        """Retrieves file-names of selected bad-channel files. Triggers when new files are loaded or reset-bad-channels button is clicked. The latter removes selected files.

        Args:
            list_selected_file_names (list): List of strings of selected bad-channel file-names.
            reset_bad_channels (int): Num clicks on reset-bad-channels button.

        Returns:
            tuple(list, list): Both lists contain strings of selected model-output file-names.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'reset-bad-channels' in trigger:
            print('Resetting loaded bad-channels')
            return [], None
        elif list_selected_file_names:
            print('Selected files: {}'.format(list_selected_file_names))
            return _get_list_for_displaying(list_selected_file_names), list_selected_file_names

    # Add loaded bad channels to dropdown menu callback
    @app.callback(
        Output('bad-channels-dropdown', 'value', allow_duplicate=True),
        Input('upload-bad-channels', 'filename'),
        State('bad-channels-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _load_bad_channel_files(loaded_bad_channel_files, current_selected_bad_channels):
        if loaded_bad_channel_files:
            for file_name in loaded_bad_channel_files:
                if '.txt' in file_name:
                    loaded_bad_channels = parse_bad_channels_file(file_name)

                    globals.bad_channels[file_name] = loaded_bad_channels
                    current_selected_bad_channels += loaded_bad_channels

            current_selected_bad_channels = list(set(current_selected_bad_channels))

        else:
            current_selected_bad_channels = globals.bad_channels[globals.file_name]

        # globals.raw.info['bads'] = current_selected_bad_channels

        return current_selected_bad_channels

    # Register channel click
    @app.callback(
        Output('bad-channels-dropdown', 'value', allow_duplicate=True),
        Input('EEG-graph', 'clickData'),
        [State('bad-channels-dropdown', 'value'), State('data-file', 'children'),],
        prevent_initial_call=True
    )
    def  _update_bad_channels_after_click(clickData, current_selected_bad_channels, file_name):
        """Updates bad-channel dropdown menu when channel is clicked.

        Args:
            clickData (dict): Data from latest click event.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """
        # print('Clicked bad channel')
        # print(globals.bad_channels)

        if globals.plotting_data:
            channel_index = clickData['points'][0]['curveNumber']
            if channel_index < len(globals.plotting_data['EEG']['channel_names']):
                channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                
                if channel_name not in current_selected_bad_channels:
                    current_selected_bad_channels.append(channel_name)
                    globals.bad_channels[file_name].append(channel_name)
                else:
                    current_selected_bad_channels.remove(channel_name)
                    if channel_name in globals.bad_channels[file_name]:
                        globals.bad_channels[file_name].remove(channel_name)

                # globals.raw.info['bads'] = current_selected_bad_channels

                return current_selected_bad_channels

        raise PreventUpdate

    # Update plot when bad channels changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('bad-channels-dropdown', 'value'),
        [State('EEG-graph', 'figure'), State('data-file', 'children'), State('hide-bad-channels-button', 'n_clicks'),],
        prevent_initial_call=True
    )
    def  _update_bad_channels_in_plot(current_selected_bad_channels, current_fig, file_name, hide_bad_channels):
        """Updates plot when bad channels changed.

        Args:
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """
        if globals.plotting_data:
            # print(current_selected_bad_channels)
            # print(globals.raw.info['bads'])

            patched_fig = Patch()

            new_bad_channels = [added_bad_channel for added_bad_channel in current_selected_bad_channels if added_bad_channel not in globals.raw.info['bads']]
            print(new_bad_channels)
            for channel_name in new_bad_channels:
                channel_index = globals.plotting_data['EEG']['channel_names'].index(channel_name)

                if not all(channel_name in annotation for annotation in globals.bad_channels.values()):
                    patched_fig['data'][channel_index]['marker']['color'] = c.BAD_CHANNEL_DISAGREE_COLOR
                else:
                    patched_fig['data'][channel_index]['marker']['color'] = c.BAD_CHANNEL_COLOR

                # If bad channels are currently hidden
                if hide_bad_channels % 2 != 0:
                    patched_fig['data'][channel_index]['visible'] = False

            removed_bad_channels = [removed_bad_channel for removed_bad_channel in globals.raw.info['bads'] if removed_bad_channel not in current_selected_bad_channels]
            print(removed_bad_channels)
            for channel_name in removed_bad_channels:
                channel_index = globals.plotting_data['EEG']['channel_names'].index(channel_name)

                patched_fig['data'][channel_index]['marker']['color'] = 'black'

            globals.raw.info['bads'] = current_selected_bad_channels

            return patched_fig
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

    # Update disagreed bad channels callback
    @app.callback(
        Output('hidden-output', 'children', allow_duplicate=True),
        Input('bad-channels-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _update_hide_bad_channels_button(current_selected_bad_channels):
        # If there are at least 2 lists of bad channels
        if len([annotation for annotation in globals.bad_channels if annotation]) > 1:
            disagreed_bad_channels = []

            for bad_channel in current_selected_bad_channels:
                # If a bad channel does not appear in all lists
                if sum(bad_channel in annotation for annotation in globals.bad_channels.values()) < len([annotation for annotation in globals.bad_channels.values() if annotation]):
                    disagreed_bad_channels.append(bad_channel)

            globals.disagreed_bad_channels = list(set(disagreed_bad_channels))
            # print(globals.disagreed_bad_channels)

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
