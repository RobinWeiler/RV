import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

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

    # Change color of clicked channel
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

    # Update "Hide/show bad channels" button when bad channels changed
    # @app.callback(
    #     Output('EEG-graph', 'figure', allow_duplicate=True),
    #     Input('bad-channels-dropdown', 'value'),
    #     State('EEG-graph', 'figure'),
    #     prevent_initial_call=True
    # )
    # def _update_bad_channels_hiding(current_selected_bad_channels, current_fig):
    #     """Updates "Hide/show bad channels" button when bad channels changed.

    #     Args:
    #         current_selected_bad_channels (list): List containing names of currently selected bad channels.
    #         current_fig (plotly.graph_objs.Figure): The current EEG plot.

    #     Returns:
    #         tuple(plotly.graph_objs.Figure, int): Updated EEG plot.
    #     """
    #     # print('trigger')
    #     # print(current_selected_bad_channels)

    #     if globals.plotting_data:
    #         globals.raw.info['bads'] = current_selected_bad_channels

    #         patched_fig = Patch()

    #         # If bad channels are currently hidden
    #         if not any(channel['visible'] == True and globals.plotting_data['EEG']['channel_names'][index] in current_selected_bad_channels for index, channel in enumerate(current_fig['data'])):
    #             temp = globals.plotting_data['EEG']['default_channel_colors']
    #             globals.plotting_data['EEG']['default_channel_visibility'] = globals.plotting_data['EEG']['channel_visibility']
    #             globals.plotting_data['EEG']['channel_visibility'] = temp

    #         for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
    #             channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
                
    #             if channel_name in current_selected_bad_channels:
    #                 globals.plotting_data['EEG']['default_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
    #                 globals.plotting_data['EEG']['channel_visibility'][channel_index] = False
    #                 globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = c.BAD_CHANNEL_COLOR
    #             else:
    #                 globals.plotting_data['EEG']['default_channel_colors'][channel_index] = 'black'
    #                 globals.plotting_data['EEG']['channel_visibility'][channel_index] = True
    #                 globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'black'

    #             # current_fig['data'][channel_index]['marker']['color'] = globals.plotting_data['EEG']['default_channel_colors'][channel_index]

    #         for model_index in range(len(globals.plotting_data['model'])):
    #             if globals.plotting_data['model'][model_index]['model_channels']:
    #                 for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
    #                     channel_name = globals.plotting_data['EEG']['channel_names'][channel_index]
    #                     if channel_name in globals.plotting_data['model'][model_index]['model_channels']:
    #                         globals.plotting_data['EEG']['highlighted_channel_colors'][channel_index] = 'green'

    #         # patched_fig['layout']['updatemenus'][0]['buttons'][2] = dict(
    #         #     label='Hide/show bad channels',
    #         #     method='restyle',
    #         #     args2=[{'visible': globals.plotting_data['EEG']['default_channel_visibility']}],
    #         #     args=[{'visible': globals.plotting_data['EEG']['channel_visibility']}]
    #         # ),
    #         patched_fig['layout']['updatemenus'][0]['buttons'][2]['args2'][0]['visible'] = globals.plotting_data['EEG']['default_channel_visibility']
    #         patched_fig['layout']['updatemenus'][0]['buttons'][2]['args'][0]['visible'] = globals.plotting_data['EEG']['channel_visibility']

    #         # patched_fig['layout']['updatemenus'][0]['buttons'][3] = dict(
    #         #     label='Highlight model-channels',
    #         #     method='restyle',
    #         #     args2=[{'marker.color': globals.plotting_data['EEG']['default_channel_colors']}],
    #         #     args=[{'marker.color': globals.plotting_data['EEG']['highlighted_channel_colors']}],
    #         #     visible=True if globals.plotting_data['model'] else False
    #         # ),
    #         patched_fig['layout']['updatemenus'][0]['buttons'][3]['args'][0]['marker.color'] = globals.plotting_data['EEG']['highlighted_channel_colors']
    #         patched_fig['layout']['updatemenus'][0]['buttons'][3]['args2'][0]['marker.color'] = globals.plotting_data['EEG']['default_channel_colors']

    #         return patched_fig
    #     else:
    #         raise PreventUpdate

    # Hide/show bad channels
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('hide-bad-channels-button', 'n_clicks'),
        [State('EEG-graph', 'figure'), State('bad-channels-dropdown', 'value')],
        prevent_initial_call=True
    )
    def  _update_bad_channels_after_click(hide_bad_channels, current_fig, current_selected_bad_channels):
        """Hides bad channels when pressed. Shows all channels when pressed again.

        Args:
            hide_bad_channels (dict): Num clicks on hide-bad-channels-button button.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            plotly.graph_objs.Figure: Updated EEG plot.
        """
        print('Button pressed')

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
