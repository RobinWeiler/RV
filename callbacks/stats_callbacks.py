import collections

from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from plotly.graph_objs import Figure

import numpy as np

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.stats_helperfunctions import _get_amount_annotated_clean_data, _get_clean_intervals, _get_annotated_overlap, get_clean_intervals_graph, get_power_spectrum_plot, _natural_keys
from helperfunctions.visualization_helperfunctions import _get_list_for_displaying

import globals

def register_stats_callbacks(app):

    # Toggle stats modal
    @app.callback(
        Output('stats-body', 'children'),
        Input("modal-stats", "is_open"),
        [State('data-file', 'children'), State('bad-channels-dropdown', 'value'), State('annotation-label', 'options')],
        prevent_initial_call=True
    )
    def _toggle_stats_modal(stats_modal_is_open, file_name, current_selected_bad_channels, annotation_labels):
        """Opens or closes stats modal based on relevant button clicks and loads all statistics.

        Args:
            open_stats1 (int): Num clicks on open-stats1 button.
            open_stats2 (int): Num clicks on open-stats2 button.
            file_name (string): File-name of selected recording.
            current_selected_bad_channels (list): List of strings of currently selected bad-channel names.

        Returns:
            tuple(bool, html.Div): Whether or not modal should be open, stats.
        """
        if stats_modal_is_open:
            if globals.raw:
                all_marked_annotations = get_annotations(globals.raw)
                print(all_marked_annotations)

                recording_length = globals.raw.n_times / globals.raw.info['sfreq']

                total_amount_annotated_data, total_amount_clean_data = _get_amount_annotated_clean_data(all_marked_annotations, recording_length)
                total_clean_interval_lengths, total_amount_clean_intervals = _get_clean_intervals(all_marked_annotations, recording_length, interval_length=2)
                total_amount_clean_data_percentage = (total_amount_clean_data / recording_length) * 100

                graph = get_clean_intervals_graph(total_clean_interval_lengths, recording_length)

                recording_length = round(recording_length)
                total_amount_clean_data = round(total_amount_clean_data)
                total_amount_clean_data_percentage = round(total_amount_clean_data_percentage)
                total_amount_annotated_data = round(total_amount_annotated_data)

            # Annotation stats
            annotation_stats = html.Div([
                html.H1('Annotations'),

                html.Div([
                    html.H2('Total amount of annotated data:'),
                    html.Font('{} seconds'.format(total_amount_annotated_data) if globals.raw else '-', id='#annotated-data')
                ]),
            ])

            if globals.raw:
                sorted_annotations = []
                for annotation_option in annotation_labels:
                    corresponding_annotations = [annotation for annotation in all_marked_annotations if annotation[2] == annotation_option['label']]
                    sorted_annotations.append(corresponding_annotations)

                    amount_annotated_data, _ = _get_amount_annotated_clean_data(corresponding_annotations, recording_length)
                    amount_annotated_data = round(amount_annotated_data)
                    amount_annotated_data_percentage = (amount_annotated_data / recording_length) * 100
                    amount_annotated_data_percentage = round(amount_annotated_data_percentage)

                    annotation_stats.children.append(
                        html.Div([
                            html.H2('Amount of annotated data of {}:'.format(annotation_option['label'])),
                            html.Font('{} seconds ({}% of recording)'.format(amount_annotated_data, amount_annotated_data_percentage), id='#annotated-data-{}'.format(annotation_option['label']))
                        ]),
                    )

                for annotation_index1 in range(len(sorted_annotations) - 1):
                    # print(annotation_index1)
                    if not sorted_annotations[annotation_index1]:
                        continue
                    for annotation_index2 in range(annotation_index1 + 1, len(sorted_annotations)):
                        # print(annotation_index2)
                        if not sorted_annotations[annotation_index2]:
                            continue
                        
                        amount_annotated_overlap = _get_annotated_overlap(sorted_annotations[annotation_index1], sorted_annotations[annotation_index2])
                        annotation_stats.children.append(
                            html.Div([
                                html.H2('Amount of overlap between annotations of {} and {}:'.format(sorted_annotations[annotation_index1][0][2], sorted_annotations[annotation_index2][0][2])),
                                html.Font(str(amount_annotated_overlap) + '%', id='#annotated-overlap-{}-{}'.format(sorted_annotations[annotation_index1][0][2], sorted_annotations[annotation_index2][0][2]))
                            ]),
                        )

            # Bad channel stats
            if current_selected_bad_channels:
                current_selected_bad_channels.sort(key=_natural_keys)
            bad_channel_stats = html.Div([
                html.H1('Bad channels'),

                html.Div([
                    html.H2('Current bad channels:'),
                    html.Font(_get_list_for_displaying(current_selected_bad_channels) if current_selected_bad_channels else ['-'], id='total-bad-channels')
                ]),
            ])

            if len(globals.bad_channels) > 1:
                if globals.disagreed_bad_channels:
                    globals.disagreed_bad_channels.sort(key=_natural_keys)
                stat_disagreed_bad_channels = globals.disagreed_bad_channels  # .copy()

                # for bad_channel in globals.disagreed_bad_channels:
                #     # Don't include agreed bad channels absent in current session
                #     if bad_channel not in globals.bad_channels['current session'] and all(bad_channel in annotation for annotator, annotation in globals.bad_channels.items() if annotation and annotator != 'current session'):
                #         stat_disagreed_bad_channels.remove(bad_channel) 

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
                            html.H1('General'),

                            html.Div([
                                html.H2('File name:'),
                                html.Font(file_name if globals.raw else '-', id='file-name')
                            ]),
                            html.Div([
                                html.H2('Recording length:'),
                                html.Font('{} seconds'.format(recording_length) if globals.raw else '-', id='recording-length')
                            ]),
                        ]),

                        html.Hr(),

                        # Clean stats
                        html.Div([
                            html.H1('Clean data'),

                            html.Div([
                                html.H2('Total amount of clean data left:'),
                                html.Font('{} seconds ({}% of recording)'.format(total_amount_clean_data, total_amount_clean_data_percentage) if globals.raw else '-', id='#clean-data')
                            ]),
                            html.Div([
                                html.H2('Total amount of clean intervals longer than 2 seconds:'),
                                html.Font(total_amount_clean_intervals if globals.raw else '-', id='#clean-intervals')
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
                        ]),

                        html.Hr(),

                        annotation_stats,
                        
                        html.Hr(),

                        bad_channel_stats
                    ]),

            return stats
        else:
            raise PreventUpdate

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
        trigger = [p['prop_id'] for p in callback_context.triggered][0]

        if 'EEG-graph' in trigger and (not selected_data or (not selected_data['points'])):
            raise PreventUpdate

        return _toggle_modal([open_power_spectrum, close_power_spectrum, selected_data], is_open)

    # Data selection returns power-spectrum
    @app.callback(
        [Output('power-selected-interval', 'children'), Output('power-selected-channels', 'children'), Output('flat-selected-channels', 'children'), Output('power-spectrum', 'figure')],  # Output('power-prominent-frequency', 'children')
        Input('EEG-graph', 'selectedData'),
        State("low-pass", "value"),
    )
    def _get_selected_power_spectrum(selected_data, low_pass):
        """Calculates frequency with highest power density and power-spectrum plot of selected data.

        Args:
            selected_data (dict): Data from latest selection event.

        Returns:
            tuple(string, plotly.graph_objs.Figure): String of frequency with highest power density, power-spectrum plot of selected_data.
        """
        if not selected_data or (not selected_data['points']):
            selected_range_text = '-'
            selected_channels = '-'
            flat_channel_names = '-'
            fig = Figure()
        else:
            # print(selected_data)

            # first_trace_number = selected_data['points'][0]['curveNumber']
            # print('First trace: {}'.format(first_trace_number))

            selected_range_x = selected_data['range']['x']
            if selected_range_x[0] < 0:
                selected_range_x[0] = 0
            selected_range_y = selected_data['range']['y']

            selected_channel_indices = [index for index, value in enumerate(globals.plotting_data['plot']['y_ticks']) if selected_range_y[0] <= value <= selected_range_y[1] and value >= 0]
            selected_channels = [globals.plotting_data['EEG']['channel_names'][index] for index in selected_channel_indices]
            print('Selected channels: {}'.format(selected_channels))

            index_0 = globals.viewing_raw.time_as_index(selected_range_x[0])[0]
            index_1 = globals.viewing_raw.time_as_index(selected_range_x[1])[0]

            data_subset, _ = globals.viewing_raw[selected_channels, index_0:index_1]
            print(data_subset.shape)

            # Ignore flat channels for psd and list separately
            std = np.std(data_subset, 1)
            flat_channel_indices = np.where(std == 0)[0]
            flat_channel_names = []
            for i in flat_channel_indices:
                flat_channel_names.append(selected_channels.pop(i))
            if len(flat_channel_names) < 1:
                flat_channel_names = '-'
            print('Flat channels: {}'.format(flat_channel_names))

            num_points = int(index_1 - index_0)
            print(num_points)
            if num_points < 256:
                n_fft = num_points
            else:
                n_fft = 256

            # fs = globals.viewing_raw.info['sfreq']
            # n_fft_seconds = 10
            # n_fft = int(fs*n_fft_seconds)
            # n_overlap = int(n_fft/2)

            spectrum = globals.viewing_raw.compute_psd('welch', picks=selected_channels, fmax=(low_pass + (low_pass * 0.2)), tmin=selected_range_x[0], tmax=selected_range_x[1], n_fft=n_fft)  # , n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_fft)
            freqs = spectrum.freqs[:]
            freqs = np.round(freqs, 2)
            Pxx_den = spectrum[:]
            Pxx_den *= (1e6**2)
            np.log10(np.maximum(Pxx_den, np.finfo(float).tiny), out=Pxx_den)
            Pxx_den *= 10
            Pxx_den = np.round(Pxx_den, 2)

            mean_Pxx_den = np.mean(Pxx_den, axis=0)

            fig = get_power_spectrum_plot(freqs, Pxx_den, selected_channels, mean_Pxx_den)

            selected_range_text = '{} - {} seconds'.format(round(selected_range_x[0], 1), round(selected_range_x[1], 1))
            print(selected_range_text)

        return selected_range_text, _get_list_for_displaying(selected_channels), _get_list_for_displaying(flat_channel_names), fig
