from dash.dependencies import Input, Output


def register_preprocessing_callbacks(app):
    @app.callback(
        Output("hidden-preprocessing-output", "n_clicks"),
        [Input("high-pass", "value"), Input("low-pass", "value"), Input("bad-channel-interpolation", "value")],
    )
    def _preprocessing_changed(high_pass, low_pass, bad_channel_interpolation):
        """Sets n_clicks of hidden-preprocessing-output to 1 if either high-pass, low-pass or bad-channel-interpolation changed.
        """
        print('Preprocessing changed')
        return 1

    @app.callback(
        Output("hidden-preprocessing-output", "n_clicks", allow_duplicate=True),
        Input("plot-button", "n_clicks"),
        prevent_initial_call=True
    )
    def _plot_button_pressed(plot_button):
        """Sets n_clicks of hidden-preprocessing-output to 0 when plot-button is pressed.
        """
        print('Reset bool')
        return 0
