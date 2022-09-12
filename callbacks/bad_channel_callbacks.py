import dash
from dash.dependencies import Input, Output, State

import globals


def register_bad_channel_callbacks(app):
    # Loading bad channels and all channel names into dropdown menu and clicking channel callback
    @app.callback(
        [Output('bad-channels-dropdown', 'value'), Output('bad-channels-dropdown', 'options'), Output('selected-channels-dropdown', 'options')],
        [Input('data-file', 'children'), Input('EEG-graph', 'figure'), Input('EEG-graph', 'clickData')],
        [State('bad-channels-dropdown', 'value'), State('bad-channels-dropdown', 'options')],
        prevent_initial_call=True
    )
    def _update_bad_channel_dropdown(file, figure, clickData, current_selected_bad_channels, current_available_channels):
        """Loads channel names into bad-channels-dropdown and selected-channels-dropdown. Triggers when new file is loaded, after plot is drawn, and when a trace is clicked on to mark it as bad.

        Args:
            file (string): Current file-name.
            figure (plotly.graph_objs.Figure): EEG plot.
            clickData (dict): Data from latest click event.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.
            current_available_channels (list): List of dicts of all available channel names.

        Returns:
            tuple(list, list, list): First list contains strings of selected bad-channel names. Second and third list contain dicts of all channel names.
        """
        
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if ('data-file' in trigger or 'figure' in trigger) and globals.raw:
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
            channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
            
            if channel_name not in current_selected_bad_channels:
                current_selected_bad_channels.append(channel_name)
            else:
                current_selected_bad_channels.remove(channel_name)
            
            return current_selected_bad_channels, current_available_channels, current_available_channels
        else:
            return [], [], []
