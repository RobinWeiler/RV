import dash
from dash.dependencies import Input, Output, State

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
            print(current_selected_bad_channels)

            for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
                channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                
                if channel_name in current_selected_bad_channels:
                    globals.plotting_data['EEG']['default_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
                    globals.plotting_data['EEG']['channel_visibility'][channel_index] = False
                    globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
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

        return current_fig
