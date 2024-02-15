import collections

from dash import dcc, html
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate

from plotly.graph_objs import Figure

import numpy as np

from helperfunctions.annotation_helperfunctions import get_annotations
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.stats_helperfunctions import _get_amount_annotated_clean_data, _get_clean_intervals, _get_annotated_overlap, calc_power_spectrum, get_clean_intervals_graph, get_most_prominent_freq, get_power_spectrum_plot, _natural_keys
from helperfunctions.visualization_helperfunctions import _get_list_for_displaying

import globals
import constants as c

def register_stats_callbacks():

    # Toggle stats modal
    @callback(
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
            general_stats = html.Div([
                html.H1('General'),

                html.Div([
                    html.H2('File name:'),
                    html.Font(file_name if file_name else '-', id='file-name')
                ]),
                html.Div([
                    html.H2('Recording length:'),
                    html.Font('{} seconds'.format(round(globals.plotting_data['EEG']['recording_length'])) if globals.plotting_data['EEG']['recording_length'] else '-', id='recording-length')
                ]),
            ])

            if globals.plotting_data['EEG']:
                all_marked_annotations = get_annotations(globals.raw)
                print(all_marked_annotations)

                total_amount_annotated_data, total_amount_clean_data = _get_amount_annotated_clean_data(all_marked_annotations, globals.plotting_data['EEG']['recording_length'])
                total_clean_interval_lengths, total_amount_clean_intervals = _get_clean_intervals(all_marked_annotations, globals.plotting_data['EEG']['recording_length'], interval_length=2)
                total_amount_clean_data_percentage = (total_amount_clean_data / globals.plotting_data['EEG']['recording_length']) * 100

                graph = get_clean_intervals_graph(total_clean_interval_lengths, globals.plotting_data['EEG']['recording_length'])

                total_amount_clean_data = round(total_amount_clean_data)
                total_amount_clean_data_percentage = round(total_amount_clean_data_percentage)
                total_amount_annotated_data = round(total_amount_annotated_data)

            clean_stats = html.Div([
                html.H1('Clean data'),

                html.Div([
                    html.H2('Total amount of clean data left:'),
                    html.Font('{} seconds ({}% of recording)'.format(total_amount_clean_data, total_amount_clean_data_percentage) if globals.plotting_data['EEG'] else '-', id='#clean-data')
                ]),
                html.Div([
                    html.H2('Total amount of clean intervals longer than 2 seconds:'),
                    html.Font(total_amount_clean_intervals if globals.plotting_data['EEG'] else '-', id='#clean-intervals')
                ]),
                html.Div([
                    dcc.Graph(
                        id='clean-intervals-graph',
                        figure=graph if globals.plotting_data['EEG'] else Figure(),
                        config={
                            'displayModeBar': False,
                        },
                    ),
                ]),
            ])

            # Annotation stats
            annotation_stats = html.Div([
                html.H1('Annotations'),

                html.Div([
                    html.H2('Total amount of annotated data:'),
                    html.Font('{} seconds'.format(total_amount_annotated_data) if globals.plotting_data['EEG'] else '-', id='#annotated-data')
                ]),
            ])

            if globals.plotting_data['EEG']:
                sorted_annotations = []
                for annotation_option in globals.plotting_data['annotations']['annotation_label_colors'].keys():
                    corresponding_annotations = [annotation for annotation in all_marked_annotations if annotation[2] == annotation_option]
                    sorted_annotations.append(corresponding_annotations)

                    amount_annotated_data, _ = _get_amount_annotated_clean_data(corresponding_annotations, globals.plotting_data['EEG']['recording_length'])
                    amount_annotated_data = round(amount_annotated_data)
                    amount_annotated_data_percentage = (amount_annotated_data / globals.plotting_data['EEG']['recording_length']) * 100
                    amount_annotated_data_percentage = round(amount_annotated_data_percentage)

                    annotation_stats.children.append(
                        html.Div([
                            html.H2('Amount of annotated data of {}:'.format(annotation_option)),
                            html.Font('{} seconds ({}% of recording)'.format(amount_annotated_data, amount_annotated_data_percentage), id='#annotated-data-{}'.format(annotation_option))
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

            if len(globals.plotting_data['annotations']['bad_channels']) > 1:
                if globals.plotting_data['plot']['disagreed_bad_channels']:
                    globals.plotting_data['plot']['disagreed_bad_channels'].sort(key=_natural_keys)

                bad_channel_stats.children.append(
                    html.Div([
                        html.H2('Disagreed bad channels:'),
                        html.Font(_get_list_for_displaying(globals.plotting_data['plot']['disagreed_bad_channels']) if globals.plotting_data['plot']['disagreed_bad_channels'] else ['-'], id='disagreed-bad-channels')
                    ]),
                )

                for annotator, bad_channels in globals.plotting_data['annotations']['bad_channels'].items():
                    if bad_channels:
                        bad_channels.sort(key=_natural_keys)
                    bad_channel_stats.children.append(
                        html.Div([
                            html.H2('Bad channels - {}'.format(annotator)),
                            html.Font(_get_list_for_displaying(bad_channels) if bad_channels else ['-'], id='{}-bad-channels'.format(annotator))
                        ]),
                    )

            stats = html.Div([
                general_stats,

                html.Hr(),

                clean_stats,

                html.Hr(),

                annotation_stats,
                
                html.Hr(),

                bad_channel_stats
            ])

            return stats
        else:
            raise PreventUpdate

    # Toggle power-spectrum modal
    @callback(
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
    @callback(
        [Output('power-selected-interval', 'children'), Output('power-selected-channels', 'children'), Output('power-spectrum', 'figure')],  # Output('power-prominent-frequency', 'children')
        Input('EEG-graph', 'selectedData'),
        State('EEG-graph', 'figure')
    )
    def _get_selected_power_spectrum(selectedData, current_fig):
        """Calculates frequency with highest power density and power-spectrum plot of selectedData.

        Args:
            selectedData (dict): Data from latest selection event.

        Returns:
            tuple(string, plotly.graph_objs.Figure): String of frequency with highest power density, power-spectrum plot of selectedData.
        """
        if not selectedData or (not selectedData['points']):
            selected_range = '-'
            selected_channels = '-'
            # most_prominent_freq = '-'
            fig = Figure()
        else:
            # print(selectedData)

            # first_trace_number = selectedData['points'][0]['curveNumber']
            # print('First trace: {}'.format(first_trace_number))

            selected_range = selectedData['range']['x']
            selected_range = (round(selected_range[0], 1), round(selected_range[1], 1))
            selected_range = '{} - {} seconds'.format(selected_range[0], selected_range[1])
            # print('Range: {}'.format(selected_range))

            split_dict = collections.defaultdict(list)

            for datapoint in selectedData['points']:
                split_dict[datapoint['curveNumber']].append(datapoint['customdata'])

            split_dict = dict(split_dict)

            max_length_selection = len(max(split_dict.values(), key=len))

            filtered_dict = split_dict.copy()

            for trace_number, datapoints in split_dict.items():
                if len(datapoints) < max_length_selection:
                    print('Not considering trace {}'.format(trace_number))
                    del filtered_dict[trace_number]

            selected_datapoints = list(filtered_dict.values())
            selected_traces = list(filtered_dict.keys())
            del filtered_dict

            selected_channels = []
            for trace in selected_traces:
                selected_channels.append(current_fig['data'][trace]['name'])
            print('Selected channels {}'.format(selected_channels))

            sample_rate = globals.viewing_raw.info['sfreq']

            all_Pxx_den = []

            for counter, trace in enumerate(selected_datapoints):
                # print(counter)
                f, Pxx_den = calc_power_spectrum(sample_rate, trace)

                # Log scaling
                Pxx_den /= globals.plotting_data['EEG']['scaling_factor']
                Pxx_den *= (c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS**2)
                np.log10(np.maximum(Pxx_den, np.finfo(float).tiny), out=Pxx_den)
                Pxx_den *= 10
                Pxx_den = np.round(Pxx_den, 2)

                all_Pxx_den.append(Pxx_den)

            mean_Pxx_den = np.mean(all_Pxx_den, axis=0)

            # most_prominent_freq = get_most_prominent_freq(f, mean_Pxx_den)
            # most_prominent_freq = round(most_prominent_freq)

            # print('Generating power spectra')
            fig = get_power_spectrum_plot(f, all_Pxx_den, selected_channels, mean_Pxx_den)

        return selected_range, _get_list_for_displaying(selected_channels), fig  # '{} Hz'.format(most_prominent_freq)
