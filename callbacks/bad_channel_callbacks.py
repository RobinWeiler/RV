import dash
from dash.dependencies import Input, Output, State

from helperfunctions.loading_helperfunctions import parse_bad_channels_file
import constants as c
import globals


def register_bad_channel_callbacks(app):

    # Loading bad channels and all channel names into dropdown menu and clicking channel callback
    @app.callback(
        [Output('bad-channels-dropdown', 'value'), Output('bad-channels-dropdown', 'options'), Output('selected-channels-dropdown', 'options')],
        [Input('data-file', 'children'), Input('EEG-graph', 'clickData')],
        [State('bad-channels-dropdown', 'value'), State('bad-channels-dropdown', 'options')],
        prevent_initial_call=True
    )
    def _update_bad_channel_dropdown(file, clickData, current_selected_bad_channels, current_available_channels):
        """Loads channel names into bad-channels-dropdown and selected-channels-dropdown. Triggers when new file is loaded and when a trace is clicked on to mark it as bad.

        Args:
            file (string): Current file-name.
            clickData (dict): Data from latest click event.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.
            current_available_channels (list): List of dicts of all available channel names.

        Returns:
            tuple(list, list, list): First list contains strings of selected bad-channel names. Second and third list contain dicts of all channel names.
        """
        
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)

        if 'data-file' in trigger and globals.raw:
            print('Loading bad channel dropdown menu...')

            all_channel_names = globals.raw.ch_names

            dropdown_channel_names = []
            for channel in all_channel_names:
                dropdown_channel_names.append({'label': channel, 'value': channel})

            bad_channels = globals.raw.info['bads']

            return bad_channels, dropdown_channel_names, dropdown_channel_names
        elif 'clickData' in trigger:
            # print('Clicked point: {}'.format(clickData))
        
            channel_index = clickData['points'][0]['curveNumber']
            if channel_index >= len(globals.plotting_data['EEG']['channel_names']):
                return current_selected_bad_channels, current_available_channels, current_available_channels

            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
            
            if channel_name not in current_selected_bad_channels:
                current_selected_bad_channels.append(channel_name)
            else:
                current_selected_bad_channels.remove(channel_name)
            
            return current_selected_bad_channels, current_available_channels, current_available_channels
        else:
            return [], [], []

    # Select bad-channel files
    @app.callback(
        [Output('bad-channel-files', 'children'), Output('upload-bad-channels', 'filename')],
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
            return list_selected_file_names, list_selected_file_names

    @app.callback(
        Output('bad-channels-dropdown', 'value', allow_duplicate=True),
        Input('bad-channel-files', 'children'),
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
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('bad-channels-dropdown', 'value'),
        State('EEG-graph', 'figure'),
        prevent_initial_call=True
    )
    def _update_EEG_plot_model(current_selected_bad_channels, current_fig):
        """Updates plot when bad channels changed.

        Args:
            current_selected_bad_channels (list): List containing names of currently selected bad channels.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(plotly.graph_objs.Figure, int): Updated EEG plot.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if globals.plotting_data:
            globals.raw.info['bads'] = current_selected_bad_channels
            # print(current_selected_bad_channels)

            for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
                channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                
                if channel_name in current_selected_bad_channels:
                    globals.plotting_data['EEG']['default_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR if channel_name not in globals.disagreed_bad_channels else c.BAD_CHANNEL_DISAGREE_COLOR
                    globals.plotting_data['EEG']['channel_visibility'][channel_index] = False
                    globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR if channel_name not in globals.disagreed_bad_channels else c.BAD_CHANNEL_DISAGREE_COLOR
                else:
                    globals.plotting_data['EEG']['default_channel_colors'][channel_index] = 'black'
                    globals.plotting_data['EEG']['channel_visibility'][channel_index] = True
                    globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'black'

                current_fig['data'][channel_index]['marker']['color'] = globals.plotting_data['EEG']['default_channel_colors'][channel_index]

            for model_index in range(len(globals.plotting_data['model'])):
                if globals.plotting_data['model'][model_index]['model_channels']:
                    for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
                        channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                        if channel_name in globals.plotting_data['model'][model_index]['model_channels']:
                            globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'blue'

            # current_fig['layout']['updatemenus'][0]['buttons'][2]['args2'][0]['visible'] = True
            current_fig['layout']['updatemenus'][0]['buttons'][2]['args'][0]['visible'] = globals.plotting_data['EEG']['channel_visibility']

            # current_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'] = globals.plotting_data['EEG']['highlighted_channel_colors']
            # current_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'] = globals.plotting_data['EEG']['default_channel_colors']

        return current_fig
