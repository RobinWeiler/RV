from dash import Patch
from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback, ctx, no_update
from dash.exceptions import PreventUpdate

import RV.constants as c
from RV.callbacks.utils.bad_channels_utils import bad_channel_disagrees


def register_bad_channel_callbacks():
    # Activate bad-channel marking mode when RV-mark-bad-channels-button is clicked
    clientside_callback(
        """
            function(n_clicks) {
                document.querySelector("a[data-val='select']").click()
                return window.dash_clientside.no_update       
            }
        """,
        [
            Input('RV-mark-bad-channels-button', 'n_clicks'),
            Input('RV-select-segment-button', 'n_clicks'),
        ],
        prevent_initial_call=True
    )

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True), Output('RV-main-graph-resampler', 'data', allow_duplicate=True),
            Output('RV-bad-channels-dropdown', 'value', allow_duplicate=True)
        ],
        [
            Input('RV-main-graph', 'clickData'),
            Input('RV-bad-channels-dropdown', 'value')
        ],
        [
            State('RV-plotting-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-hide-bad-channels-button', 'n_clicks'),
        ],
        prevent_initial_call=True
    )
    def recolor_bad_channels(click_data, selected_bad_channels, plotting_data, resampler, hide_bad_channels):
        """Re-colors traces clicked on by user or selected in RV-bad-channels-dropdown.
        Also updates RV-bad-channels-dropdown accordingly.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        patched_fig = Patch()

        if 'RV-main-graph' in trigger:
            channel_index = click_data['points'][0]['curveNumber']
            if channel_index >= len(plotting_data['selected_channels']):
                print('Cannot mark this trace as bad.')
                raise PreventUpdate

            channel_name = plotting_data['selected_channels'][channel_index]
            print(f'Clicked channel: {channel_name}')

            if channel_name not in selected_bad_channels:
                selected_bad_channels.append(channel_name)

                if bad_channel_disagrees(channel_name, plotting_data['bad_channels']):
                    channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
                else:
                    channel_color = c.BAD_CHANNEL_COLOR

                if resampler != None:
                    patched_fig['data'][channel_index]['marker']['color'] = channel_color
                    patched_fig['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)
                    resampler['data'][channel_index]['marker']['color'] = channel_color
                    resampler['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)
            else:
                selected_bad_channels.remove(channel_name)

                if resampler != None:
                    patched_fig['data'][channel_index]['marker']['color'] = 'black'
                    resampler['data'][channel_index]['marker']['color'] = 'black'

        elif 'RV-bad-channels-dropdown' in trigger:
            for channel_index, channel_name in enumerate(plotting_data['selected_channels']):
                if channel_name in selected_bad_channels:
                    if bad_channel_disagrees(channel_name, plotting_data['bad_channels']):
                        channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
                    else:
                        channel_color = c.BAD_CHANNEL_COLOR

                    if resampler != None:
                        patched_fig['data'][channel_index]['marker']['color'] = channel_color
                        patched_fig['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)
                        resampler['data'][channel_index]['marker']['color'] = channel_color
                        resampler['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)
                else:
                    if resampler != None:
                        patched_fig['data'][channel_index]['marker']['color'] = 'black'
                        patched_fig['data'][channel_index]['visible'] = True
                        resampler['data'][channel_index]['marker']['color'] = 'black'
                        resampler['data'][channel_index]['visible'] = True

        if resampler is None:
            return no_update, no_update, selected_bad_channels

        return patched_fig, Serverside(resampler), selected_bad_channels

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-channel-selection-graph', 'figure', allow_duplicate=True),
        ],
        Input('RV-bad-channels-dropdown', 'value'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-channel-selection-graph', 'figure'),
        ],
        prevent_initial_call=True
    )
    def update_bad_channels(selected_bad_channels, raw, plotting_data, channel_selection_fig):
        """Updates mne.io.Raw object and RV-channel-selection-graph according to changes to RV-bad-channels-dropdown.
        """
        if raw is None:
            raise PreventUpdate

        new_bad_channels = [new_bad_channel for new_bad_channel in selected_bad_channels if new_bad_channel not in raw.info['bads']]
        removed_bad_channels = [removed_bad_channel for removed_bad_channel in raw.info['bads'] if removed_bad_channel not in selected_bad_channels]
        if len(new_bad_channels) == 0 and len(removed_bad_channels) == 0:
            raise PreventUpdate

        if channel_selection_fig != None:
            patched_channel_fig = Patch()
        else:
            patched_channel_fig = no_update

        for channel_name in new_bad_channels:
            if channel_name in plotting_data['selected_channels']:
                if channel_selection_fig != None:
                    if bad_channel_disagrees(channel_name, plotting_data['bad_channels']):
                        channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
                    else:
                        channel_color = c.BAD_CHANNEL_COLOR

                    # channel_index in RV-channel-selection-graph is based on original channel order
                    channel_index = raw.ch_names.index(channel_name)
                    patched_channel_fig['data'][channel_index]['marker']['color'] = channel_color

        for channel_name in removed_bad_channels:
            if channel_name in plotting_data['selected_channels']:
                if channel_selection_fig != None:
                    channel_index = raw.ch_names.index(channel_name)
                    patched_channel_fig['data'][channel_index]['marker']['color'] = 'black'

        raw.info['bads'] = selected_bad_channels

        return Serverside(raw), patched_channel_fig

    @callback(
        [
            Output('RV-plotting-data', 'data', allow_duplicate=True),
            Output('RV-bad-channels-dropdown', 'value', allow_duplicate=True)
        ],
        Input('RV-bad-channel-file-selection-dropdown', 'value'),
        [
            State('RV-plotting-data', 'data'),
            State('RV-bad-channels-dropdown', 'value')
        ],
        prevent_initial_call=True
    )
    def load_bad_channel_files(selected_bad_channel_files, plotting_data, selected_bad_channels):
        """Loads bad channels from bad-channel files selected in RV-bad-channel-file-selection-dropdown into RV-bad-channels-dropdown.
        Currently supported file formats are .txt.
        """
        if selected_bad_channel_files:
            for file in selected_bad_channel_files:
                with open(file, 'r') as f:
                    line = f.readline().rstrip(',')
                    strings = line.split(',')

                # Remove leading/trailing whitespace from each string
                loaded_bad_channels = [s.strip() for s in strings]
                # Filter out empty strings
                loaded_bad_channels = [s for s in loaded_bad_channels if s]

                print(f'Loaded bad channels: {loaded_bad_channels}')

                plotting_data['bad_channels'][str(file)] = loaded_bad_channels

                for channel_name in loaded_bad_channels:
                    if channel_name not in selected_bad_channels:
                        selected_bad_channels.append(channel_name)

            return plotting_data, selected_bad_channels

        raise PreventUpdate

    @callback(
        Output('RV-hide-bad-channels-button', 'disabled'),
        [
            Input('RV-bad-channels-dropdown', 'value'), Input('RV-channel-selection-dropdown', 'value')
        ],
        State('RV-plotting-data', 'data'),
        prevent_initial_call=True
    )
    def disable_hide_bad_channels_button_callback(selected_bad_channels, current_plotted_channels, plotting_data):
        """Disables/enables hide-bad-channels-button.
        Triggered when selected bad channels (RV-bad-channels-dropdown) or plotted channels (RV-channel-selection-dropdown) change.
        """
        if not selected_bad_channels:
            return True
        # if all channels are plotted
        elif not current_plotted_channels:
            return False
        # if at least one bad channels is being plotted
        elif any(channel in current_plotted_channels for channel in selected_bad_channels):
            return False
        else:
            return True

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True), Output('RV-main-graph-resampler', 'data', allow_duplicate=True)
        ],
        Input('RV-hide-bad-channels-button', 'n_clicks'),
        [ 
            State('RV-plotting-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-bad-channels-dropdown', 'value')
        ],
        prevent_initial_call=True
    )
    def  hide_bad_channels_button_callback(hide_bad_channels, plotting_data, resampler, selected_bad_channels):
        """Hides bad channels when 'RV-hide-bad-channels-button' is pressed. Shows all channels when pressed again.
        """
        patched_fig = Patch()

        for channel_name in selected_bad_channels:
            if channel_name in plotting_data['selected_channels']:
                channel_index = plotting_data['selected_channels'].index(channel_name)

                # visible = True if amount of clicks on RV-hide-bad-channels-button is even, otherwise False
                patched_fig['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)
                resampler['data'][channel_index]['visible'] = (hide_bad_channels % 2 == 0)

        return patched_fig, Serverside(resampler)
