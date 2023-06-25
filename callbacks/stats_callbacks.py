import collections

from dash.dependencies import Input, Output, State

from plotly.graph_objs import Figure

import numpy as np

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.stats_helperfunctions import calc_stats, calc_power_spectrum, get_clean_intervals_graph, get_most_prominent_freq, get_power_spectrum_plot

import globals

def register_stats_callbacks(app):

    # Toggle stats modal
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

    # Toggle power-spectrum modal
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

    # Data selection returns power-spectrum
    @app.callback(
        [Output('selected-data', 'children'), Output('power-spectrum', 'figure')],
        [Input('EEG-graph', 'selectedData')]
    )
    def _get_selected_power_spectrum(selectedData):
        """Calculates frequency with highest power density and power-spectrum plot of selectedData.

        Args:
            selectedData (dict): Data from latest selection event.

        Returns:
            tuple(string, plotly.graph_objs.Figure): String of frequency with highest power density, power-spectrum plot of selectedData.
        """
        if not selectedData or (not selectedData['points']):
            most_prominent_freq = '-'
            fig = Figure()
        else:
            # print(selectedData)
            # selected_data = []

            trace_number = selectedData['points'][0]['curveNumber']
            # print('First trace: {}'.format(trace_number))

            selected_range = selectedData['range']
            print('Range: {}'.format(selected_range))

            split_dict = collections.defaultdict(list)

            for datapoint in selectedData['points']:
                split_dict[datapoint['curveNumber']].append(datapoint['customdata'])

            selected_traces_list = list(split_dict.values())

            sample_rate = globals.viewing_raw.info['sfreq']

            all_Pxx_den = []

            for counter, trace in enumerate(selected_traces_list):
                # print(counter)
                f, Pxx_den = calc_power_spectrum(sample_rate, trace)
                all_Pxx_den.append(Pxx_den)

            mean_Pxx_den = np.mean(all_Pxx_den, axis=0)

            most_prominent_freq = get_most_prominent_freq(f, mean_Pxx_den)
            most_prominent_freq = round(most_prominent_freq, 2)

            fig = get_power_spectrum_plot(f, mean_Pxx_den)

        return (str(most_prominent_freq) + ' Hz'), fig