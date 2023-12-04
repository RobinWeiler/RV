import collections

from dash import dcc, html
from dash.dependencies import Input, Output, State

from plotly.graph_objs import Figure

import numpy as np

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.stats_helperfunctions import calc_stats, calc_power_spectrum, get_clean_intervals_graph, get_most_prominent_freq, get_power_spectrum_plot, _natural_keys
from helperfunctions.visualization_helperfunctions import _get_list_for_displaying

import globals

def register_stats_callbacks(app):

    # Toggle stats modal
    @app.callback(
        Output('stats-body', 'children'),
        [Input("open-stats", "n_clicks"), Input("open-stats-2", "n_clicks")],
        [State('data-file', 'children'), State('bad-channels-dropdown', 'value')],
        prevent_initial_call=True
    )
    def _toggle_stats_modal(open_stats_1, open_stats_2, file_name, current_selected_bad_channels):
        """Opens or closes stats modal based on relevant button clicks and loads all statistics.

        Args:
            open_stats1 (int): Num clicks on open-stats1 button.
            open_stats2 (int): Num clicks on open-stats2 button.
            file_name (string): File-name of selected recording.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            tuple(bool, html.Div): Whether or not modal should be open, stats.
        """
        if globals.raw:
            marked_annotations = get_annotations(globals.raw)

            recording_length = globals.raw.n_times / globals.raw.info['sfreq']

            amount_clean_data, amount_clean_intervals, clean_interval_lengths, amount_annotated_data, amount_annotated_overlap = calc_stats(marked_annotations, recording_length)

            graph = get_clean_intervals_graph(clean_interval_lengths, recording_length)

            recording_length = round(recording_length, 2)
            amount_clean_data = round(amount_clean_data, 2)
            amount_annotated_data = round(amount_annotated_data, 2)
            amount_annotated_overlap = round(amount_annotated_overlap, 2)

        # Bad channel stats
        if current_selected_bad_channels:
            current_selected_bad_channels.sort(key=_natural_keys)
        bad_channel_stats = html.Div([
            html.Div([
                html.H2('Current bad channels:'),
                html.Font(_get_list_for_displaying(current_selected_bad_channels) if current_selected_bad_channels else ['-'], id='total-bad-channels')
            ]),
        ])

        if len(globals.bad_channels) > 1:
            if globals.disagreed_bad_channels:
                globals.disagreed_bad_channels.sort(key=_natural_keys)
            stat_disagreed_bad_channels = globals.disagreed_bad_channels.copy()

            for bad_channel in globals.disagreed_bad_channels:
                # Don't include agreed bad channels absent in current session
                if bad_channel not in globals.bad_channels['current session'] and all(bad_channel in annotation for annotator, annotation in globals.bad_channels.items() if annotation and annotator != 'current session'):
                    stat_disagreed_bad_channels.remove(bad_channel) 

            bad_channel_stats.children.append(
                html.Div([
                    html.H2('Disagreed bad channels:'),
                    html.Font(_get_list_for_displaying(stat_disagreed_bad_channels) if stat_disagreed_bad_channels else ['-'], id='disagreed-bad-channels')
                ]),
            )

            for annotator, bad_channels in globals.bad_channels.items():
                if bad_channels:
                    bad_channels.sort(key=_natural_keys)
                bad_channel_stats.children.append(
                    html.Div([
                        html.H2('Bad channels - {}'.format(annotator)),
                        html.Font(_get_list_for_displaying(bad_channels) if bad_channels else ['-'], id='{}-bad-channels'.format(annotator))
                    ]),
                )

        stats = html.Div([
                    # General info
                    html.Div([
                        html.H2('File name:'),
                        html.Font([file_name if globals.raw else '-'], id='file-name')
                    ]),
                    html.Div([
                        html.H2('Recording length (in seconds):'),
                        html.Font([recording_length if globals.raw else '-'], id='recording-length')
                    ]),
                    html.Hr(),

                    # Clean stats
                    html.Div([
                        html.H2('Amount of clean data left (in seconds):'),
                        html.Font([amount_clean_data if globals.raw else '-'], id='#clean-data')
                    ]),
                    html.Div([
                        html.H2('Amount of clean intervals longer than 2 seconds:'),
                        html.Font([amount_clean_intervals if globals.raw else '-'], id='#clean-intervals')
                    ]),
                    html.Div([
                        dcc.Graph(
                            id='clean-intervals-graph',
                            figure=graph if globals.raw else Figure(),
                            config={
                                'displayModeBar': False,
                            },
                        ),
                    ]),
                    html.Hr(),

                    # Annotation stats
                    html.Div([
                        html.H2('Total amount of annotated data (in seconds):'),
                        html.Font([amount_annotated_data if globals.raw else '-'], id='#annotated-data')
                    ]),
                    html.Div([
                        html.H2('Amount of overlap between annotations (in seconds):'),
                        html.Font([amount_annotated_overlap if globals.raw else '-'], id='#annotated-overlap')
                    ]),
                    
                    html.Hr(),

                    bad_channel_stats
                ]),

        return stats

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
