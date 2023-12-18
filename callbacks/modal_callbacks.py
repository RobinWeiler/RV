import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from helperfunctions.modal_helperfunctions import _toggle_modal


def register_modal_callbacks(app):
    # Toggle preprocessing modal
    @app.callback(
        Output("modal-file", "is_open"),
        [Input("open-file", "n_clicks"), Input("close-file", "n_clicks"), Input('plot-button', 'n_clicks'), Input('cancel-plot-button', 'n_clicks')],
        [State("modal-file", "is_open")],
    )
    def _toggle_file_modal(open_file, close_file, plot_button, cancel_plot_button, is_open):
        """Opens or closes open-file modal based on relevant button clicks.

        Args:
            open_file (int): Num clicks on open-file button.
            close_file (int): Num clicks on close-file button.
            plot_button (int): Num clicks on plot button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'plot-button' in trigger and not plot_button:
            raise PreventUpdate

        return _toggle_modal([open_file, close_file, plot_button, cancel_plot_button], is_open)

    # Toggle stats modal
    @app.callback(
        Output("modal-stats", "is_open"),
        [Input("open-stats", "n_clicks"), Input("open-stats-2", "n_clicks"), Input("close-stats", "n_clicks")],
        [State("modal-stats", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_stats_modal(open_stats, open_stats_2, close_stats, is_open):
        """Opens or closes stats modal based on relevant button clicks.

        Args:
            open_stats (int): Num clicks on open-stats button.
            open_stats_2 (int): Num clicks on open-stats-2 button.
            close_stats (int): Num clicks on close-stats button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_stats, close_stats, open_stats_2], is_open)

    # Toggle help modal
    @app.callback(
        Output("modal-help", "is_open"),
        [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
        [State("modal-help", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_help_modal(open_help, close_help, is_open):
        """Opens or closes help modal based on relevant button clicks.

        Args:
            open_help (int): Num clicks on open-help button.
            close_help (int): Num clicks on close-help button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_help, close_help], is_open)

    # Toggle quit modal
    @app.callback(
        Output("modal-quit", "is_open"),
        [Input("quit-button", "n_clicks"), Input('cancel-quit-button', 'n_clicks')],
        [State("modal-quit", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_quit_modal(quit_button, cancel_quit_button, is_open):
        """Opens or closes quit modal based on relevant button clicks.

        Args:
            quit_button (int): Num clicks on quit button.
            cancel_quit_button (int): Num clicks on cancel-quit button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([quit_button, cancel_quit_button], is_open)

    # Toggle confirm-replot modal
    @app.callback(
        Output("modal-confirm-replot", "is_open"),
        [Input("confirm-plot-button", "n_clicks"), Input('cancel-plot-button', 'n_clicks'), Input('plot-button', 'n_clicks')],
        [State("modal-confirm-replot", "is_open"), State("hidden-bandpass-changed", "n_clicks")],
        prevent_initial_call=True
    )
    def _toggle_overwrite_modal(confirm_replot_button, cancel_replot_button, plot_button, is_open, bandpass_changed):
        """Opens or closes confirm-replot modal based on relevant button clicks.

        Args:
            confirm_replot_button (int): Num clicks on confirm-plot-button.
            cancel_replot_button (int): Num clicks on cancel-plot-button.
            plot_button (int): Num clicks on plot-button.
            is_open (bool):  Whether or not modal-confirm-replot is currently open.

        Returns:
            bool: Whether or not modal-confirm-replot should now be open.
        """
        if not bandpass_changed:
            raise PreventUpdate

        return _toggle_modal([confirm_replot_button, cancel_replot_button, plot_button], is_open)
