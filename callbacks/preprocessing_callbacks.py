from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import globals


def register_preprocessing_callbacks(app):
    @app.callback(
        Output("hidden-bandpass-changed", "n_clicks"),
        [Input("high-pass", "value"), Input("low-pass", "value")],
        State('plot-button', 'n_clicks')
    )
    def _bandpass_changed(high_pass, low_pass, plot_button):
        """Sets n_clicks of hidden-bandpass-changed to 1 if either high-pass or low-pass changed.
        """
        if not plot_button:
            raise PreventUpdate

        if (not high_pass or float(high_pass) == globals.raw.info['highpass']) and (not low_pass or float(low_pass) == globals.raw.info['lowpass']):
            return 0

        # print('Bandpass changed')
        return 1

    @app.callback(
        Output("hidden-bandpass-changed", "n_clicks", allow_duplicate=True),
        Input("confirm-plot-button", "n_clicks"),
        prevent_initial_call=True
    )
    def _confirm_plot_button_pressed(plot_button):
        """Sets n_clicks of hidden-bandpass-changed to 0 when confirm-plot-button is pressed.
        """
        # print('Reset bool')
        return 0

    @app.callback(
        Output("plot-button", "n_clicks"),
        Input('upload-file', 'filename'),
        prevent_initial_call=True
    )
    def _reset_plot_button(selected_file_name):
        """Sets n_clicks of plot-button to 0 when new file is loaded.
        """
        return 0
