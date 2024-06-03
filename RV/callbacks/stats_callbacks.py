from dash import html
from dash_extensions.enrich import Output, Input, State, callback
from dash.exceptions import PreventUpdate

from RV.callbacks.utils.stats_utils import get_annotation_stats, get_bad_channel_info


def register_stats_callbacks():
    @callback(
        Output({'type': 'modal', 'modal': 'RV-stats'}, 'is_open', allow_duplicate=True),
        Input('RV-open-stats-button-2', 'n_clicks'),
        prevent_initial_call=True
    )
    def open_stats_from_settings(open_stats_button_2):
        """Opens RV-stats modal when RV-open-stats-button-2 is clicked.
        """
        if open_stats_button_2:
            return True

        raise PreventUpdate

    @callback(
        Output({'type': 'modal', 'modal': 'RV-settings'}, 'is_open', allow_duplicate=True),
        Input({'type': 'modal', 'modal': 'RV-stats'}, 'is_open'),
        [
            State({'type': 'modal', 'modal': 'RV-settings'}, 'is_open'),
            State({'type': 'open-button', 'modal': 'RV-stats'}, 'n_clicks_timestamp'),
            State('RV-open-stats-button-2', 'n_clicks_timestamp')
        ],
        prevent_initial_call=True
    )
    def toggle_settings_modal_stats(stats_is_open, settings_is_open, open_stats_button_timestamp, open_stats_button_2_timestamp):
        """Toggles RV-settings modal opposite to RV-stats modal when RV-stats modal was opened through RV-open-stats-button-2.
        """
        if stats_is_open and settings_is_open:
            return False
        elif open_stats_button_2_timestamp > open_stats_button_timestamp:
            return True
        else:
            raise PreventUpdate

    @callback(
        Output('RV-stats-modal-body', 'children'),
        Input({'type': 'modal', 'modal': 'RV-stats'}, 'is_open'),
        [
            State('RV-raw', 'data'), State('RV-plotting-data', 'data'),
            State('RV-file-selection-dropdown', 'value'),
            State('RV-bad-channels-dropdown', 'value'), State('RV-annotation-label', 'options')
        ],
        prevent_initial_call=True
    )
    def get_stats(stats_modal_is_open, raw, plotting_data, selected_file, selected_bad_channels, annotation_label_options):
        """Generates html.Div with statistics about annotated EEG data to be displayed in RV-stats modal.
        """
        if stats_modal_is_open and raw != None:
            general_info = html.Div([
                html.H1('General info'),

                html.H2('File:'),
                html.Span(selected_file),

                html.H2('Recording length:'),
                html.Span(f"{round(plotting_data['recording_length'], 1)} seconds")
            ])

            annotation_stats = get_annotation_stats(raw, plotting_data['recording_length'], [annotation_label['value'] for annotation_label in annotation_label_options])

            bad_channel_stats = get_bad_channel_info(selected_bad_channels, plotting_data['bad_channels'])

            stats = html.Div([
                general_info,
                html.Hr(),

                annotation_stats,
                html.Hr(),

                bad_channel_stats,
                html.Hr(),
            ])

            return stats

        raise PreventUpdate
