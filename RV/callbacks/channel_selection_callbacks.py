from dash_extensions.enrich import Output, Input, State, callback
from dash.exceptions import PreventUpdate

from RV.callbacks.utils.channel_selection_utils import get_10_20_channels


def register_channel_selection_callbacks():
    @callback(
        Output('RV-channel-selection-dropdown', 'value', allow_duplicate=True),
        Input('RV-channel-selection-graph', 'selectedData'),
        prevent_initial_call=True
    )
    def update_selected_channels(selected_data):
        """Updates selected channels in RV-channel-selection-dropdown when datapoints are selected in RV-channel-selection-graph.
        """
        selected_channels = []

        if selected_data:
            for selected_channel in selected_data['points']:
                selected_channels.append(selected_channel['customdata'])

        print(selected_channels)

        return selected_channels

    @callback(
        Output('RV-channel-selection-dropdown', 'value', allow_duplicate=True),
        Input('RV-10-20-channels-button', 'n_clicks'),
        State('RV-raw', 'data'),
        prevent_initial_call=True
    )
    def select_10_20_channels(ten_twenty_button, raw):
        """Selects 10-20 channels in RV-channel-selection-dropdown (if possible). Triggered by RV-10-20-channels-button.
        """
        if ten_twenty_button and raw != None:
            return get_10_20_channels(raw.ch_names)

        raise PreventUpdate
