from dash_extensions.enrich import Output, Input, State, callback
from dash.exceptions import PreventUpdate

from RV.callbacks.utils.topomap_utils import get_animated_topomap_fig


def register_topomap_callbacks():
    @callback(
        [
            Output('RV-topomap-graph', 'figure'),
            Output('RV-topomap-graph', 'style')
        ],
        Input('RV-topomap-button', 'n_clicks'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-main-graph', 'selectedData'),
            State('RV-topomap-sampling-rate-input', 'value'),
            State('RV-topomap-bad-channel-interpolation', 'value')
        ],
        prevent_initial_call=True
    )
    def get_selected_segment_topomap_animation(compute_topomap, raw, plotting_data, selected_data, sampling_rate, interpolate_bad_channels):
        """Generates animation of topomap from timeframe selected in RV-main-graph. Triggered by RV-topomap-button.
        """
        if compute_topomap:
            if raw.info['dig'] is None:
                raise Exception("Cannot compute topomap without channel coordinates in raw.info['dig'].") 

            selected_interval_x = selected_data['range']['x']
            if selected_interval_x[0] < 0:
                selected_interval_x[0] = 0
            if selected_interval_x[1] > plotting_data['recording_length']:
                selected_interval_x[1] = plotting_data['recording_length']

            resampled_raw = raw.copy()
            if sampling_rate:
                resampled_raw.resample(sampling_rate)

            animated_power_topomap_fig = get_animated_topomap_fig(resampled_raw, selected_interval_x[0], selected_interval_x[1], interpolate_bad_channels)

            return animated_power_topomap_fig, {}

        raise PreventUpdate
