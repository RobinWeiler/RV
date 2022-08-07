from dash.dependencies import Input, Output, State

from plotly.graph_objs import Figure

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.stats_helperfunctions import calc_stats, get_clean_intervals_graph
from helperfunctions.visualization_helperfunctions import get_channel_locations_plot

import globals


def _toggle_modal(list_button_clicks, is_open):
    """Indicates whether modal should be open or closed based on relevant button clicks.

    Args:
        list_button_clicks (list of ints): List of clicked buttons.
        is_open (bool): Whether or not modal is currently open.

    Returns:
        bool: Whether or not modal should now be open.
    """
    for button in list_button_clicks:
        if button:
            return not is_open
    return is_open


def register_modal_callbacks(app):
    # Open-file modal callback
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

    # Select-channels modal callback
    @app.callback(
        [Output("modal-channel-select", "is_open"), Output("channel-topography", "figure")],
        [Input("open-channel-select", "n_clicks"), Input("close-channel-select", "n_clicks")],
        [State("modal-channel-select", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_channel_selection_modal(open_channel_select, close_channel_select, is_open):
        """Opens or closes channel-select modal based on relevant button clicks and loads channel-topography plot if available.

        Args:
            open_channel_select (int): Num clicks on open-channel-select button.
            close_channel_select (int): Num clicks on close-channel-select button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            tuple(bool and plotly.graph_objs.Figure): Bool whether or not modal should now be open. Figure with channel-topography plot.
        """
        topography_plot = Figure()

        if globals.raw:
            topography_plot = get_channel_locations_plot(globals.raw)
        else:
            topography_plot = Figure()

        if open_channel_select or close_channel_select:
            return not is_open, topography_plot
        return is_open, topography_plot

    # Save-file modal callback
    @app.callback(
        Output("modal-save", "is_open"),
        [Input("open-save", "n_clicks"), Input("close-save", "n_clicks"), Input('save-button', 'n_clicks'), Input('open-overwrite-button', 'n_clicks')],
        [State("modal-save", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_save_modal(open_save, close_save, save_button, open_overwrite_button, is_open):
        """Opens or closes save modal based on relevant button clicks.

        Args:
            open_save (int): Num clicks on open-save button.
            close_save (int): Num clicks on close-save button.
            save_button (int): Num clicks on save button.
            open_overwrite_button (int): Num clicks on open-overwrite button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_save, close_save, save_button, open_overwrite_button], is_open)

    # Overwrite_file modal callback
    @app.callback(
        Output("modal-overwrite", "is_open"),
        [Input("open-overwrite-button", "n_clicks"), Input('cancel-overwrite-button', 'n_clicks'), Input('overwrite-button', 'n_clicks')],
        [State("modal-overwrite", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_overwrite_modal(open_overwrite_button, cancel_overwrite_button, overwrite_button, is_open):
        """Opens or closes overwrite modal based on relevant button clicks.

        Args:
            open_overwrite_button (int): Num clicks on open-overwrite button.
            cancel_overwrite_button (int): Num clicks on cancel-overwrite button.
            overwrite_button (int): Num clicks on overwrite button.
            is_open (bool):  Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_overwrite_button, cancel_overwrite_button, overwrite_button], is_open)

    # Stats modal callback
    @app.callback(
        [
            Output("modal-stats", "is_open"), Output('file-name', 'children'), 
            Output('recording-length', 'children'), Output('#noisy-data', 'children'), 
            Output('#clean-data', 'children'), Output('#clean-intervals', 'children'), 
            Output('clean-intervals-graph', 'figure')
        ],
        [Input("open-stats", "n_clicks"), Input("open-stats-2", "n_clicks"), Input("close-stats", "n_clicks")],
        [State('data-file', 'children'), State("modal-stats", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_stats_modal(open_stats1, open_stats2, close_stats, loaded_file_name, is_open):
        """Opens or closes stats modal based on relevant button clicks and loads all statistics.

        Args:
            open_stats1 (int): Num clicks on open-stats1 button.
            open_stats2 (int): Num clicks on open-stats2 button.
            close_stats (int): Num clicks on close-stats button.
            loaded_file_name (string): File-name of selected recording.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            tuple(bool, string, float, float, float, int, plotly.graph_objs.Figure): Whether or not modal should now be open, file name, recording length, length of annotated data, length of un-annotated data, num un-annotated intervals longer than 2 seconds, histogram of un-annotated interval lengths.
        """
        marked_annotations = get_annotations(globals.raw)

        recording_length = globals.raw.n_times / globals.raw.info['sfreq']

        amount_noisy_data, amount_clean_data, amount_clean_intervals, clean_interval_lengths = calc_stats(marked_annotations, recording_length)

        graph = get_clean_intervals_graph(clean_interval_lengths, recording_length)

        recording_length = round(recording_length, 2)
        amount_noisy_data = round(amount_noisy_data, 2)
        amount_clean_data = round(amount_clean_data, 2)

        if open_stats1 or open_stats2 or close_stats:
            return not is_open, loaded_file_name, recording_length, amount_noisy_data, amount_clean_data, amount_clean_intervals, graph
        return is_open, loaded_file_name, recording_length, amount_noisy_data, amount_clean_data, amount_clean_intervals, graph

    # Open-power-spectrum modal callback
    @app.callback(
        Output("modal-power-spectrum", "is_open"),
        [Input("open-power-spectrum", "n_clicks"), Input("close-power-spectrum", "n_clicks"), Input('EEG-graph', 'selectedData')],
        [State("modal-power-spectrum", "is_open")],
        prevent_initial_call=True
    )
    def _toggle_power_spectrum_modal(open_power_spectrum, close_power_spectrum, selected_data, is_open):
        """Opens or closes power-spectrum modal based on relevant button clicks.

        Args:
            open_power_spectrum (int): Num clicks on open-power-spectrum button.
            close_power_spectrum (int): Num clicks on close-power-spectrum button.
            selected_data (dict): Data from latest select event.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Whether or not modal should now be open.
        """
        return _toggle_modal([open_power_spectrum, close_power_spectrum, selected_data], is_open)

    # Help modal callback
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

    # Quit modal callback
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
