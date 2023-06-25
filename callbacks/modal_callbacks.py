from dash.dependencies import Input, Output, State

from helperfunctions.modal_helperfunctions import _toggle_modal


def register_modal_callbacks(app):
    # Toggle preprocessing modal
    @app.callback(
        Output("modal-file", "is_open"),
        [Input("open-file", "n_clicks"), Input("close-file", "n_clicks"), Input('plot-button', 'n_clicks')],
        [State("modal-file", "is_open")],
    )
    def _toggle_file_modal(open_file, close_file, plot_button, is_open):
        """Opens or closes open-file modal based on relevant button clicks.

        Args:
            open_file (int): Num clicks on open-file button.
            close_file (int): Num clicks on close-file button.
            plot_button (int): Num clicks on plot button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_file, close_file, plot_button], is_open)

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
