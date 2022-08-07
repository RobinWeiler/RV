from dash.dependencies import Input, Output

import globals


def register_bad_channel_callbacks(app):
    # Saving bad channels when marked callback
    @app.callback(
        Output('chosen-bad-channels', 'children'),
        Input('bad-channels-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _update_bad_channels(list_bad_channels):
        """Adds selected bad channels to current Raw object. Triggers when new bad channel is selected.

        Args:
            list_bad_channels (list): List of strings of selected bad channels.
        """
        # print(list_bad_channels)

        if globals.raw:
            globals.raw.info['bads'] = list_bad_channels

            # print('Saving annotated bad channels')
            # quick_save(raw)

    # Loading bad channels and all channel names into dropdown menu callback
    @app.callback(
        [Output('bad-channels-dropdown', 'value'), Output('bad-channels-dropdown', 'options'), Output('selected-channels-dropdown', 'options')],
        [Input('data-file', 'children'), Input('EEG-graph', 'figure')]
    )
    def _update_bad_channel_dropdown(file, fig):
        """Loads channel names into bad-channels-dropdown and selected-channels-dropdown. Triggers when new file is loaded and after plot is drawn.

        Args:
            file (string): Current file-name.
            fig (plotly.graph_objs.Figure): EEG plot.

        Returns:
            tuple(list, list, list): First list contains strings of selected bad-channel names. Second and third list contain strings of all channel names.
        """

        if (file or fig) and globals.raw:
            print('Loading bad channel dropdown menu...')

            all_channel_names = globals.raw.ch_names

            dropdown_channel_names = []
            for channel in all_channel_names:
                dropdown_channel_names.append({'label': channel, 'value': channel})

            bad_channels = globals.raw.info['bads']

            return bad_channels, dropdown_channel_names, dropdown_channel_names
        else:
            return [], [], []
