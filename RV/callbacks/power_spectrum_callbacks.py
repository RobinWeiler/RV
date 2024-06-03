import multiprocessing

from dash_extensions.enrich import Output, Input, State, callback
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import numpy as np

import RV.constants as c
from RV.callbacks.utils.power_spectrum_utils import get_power_spectrum_plot


def register_power_spectrum_callbacks():
    # This callback is already registered in bad_channels_callbacks.py;
    # RV-select-segment-button and RV-mark-bad-channels-button are clicking the same hidden modebar button
    # clientside_callback(
    #     """
    #         function(n_clicks) {
    #             document.querySelector("a[data-val='select']").click()
    #             return window.dash_clientside.no_update       
    #         }
    #     """,
    #     Input('RV-select-segment-button', 'n_clicks'),
    #     prevent_initial_call=True
    # )

    @callback(
        [
            Output({'type': 'modal', 'modal': 'RV-psd'}, 'is_open', allow_duplicate=True),
            Output('RV-topomap-graph', 'figure', allow_duplicate=True),
            Output('RV-topomap-graph', 'style', allow_duplicate=True)
        ],
        Input('RV-main-graph', 'selectedData'),
        prevent_initial_call=True
    )
    def open_psd_modal(selected_data):
        """Opens RV-psd modal when data is selected in RV-main-graph.
        Also resets RV-topomap-graph.
        """
        if selected_data['points']:
            return True, go.Figure(), {'display': 'none'}

        raise PreventUpdate

    @callback(
        [
            Output('RV-selected-interval-psd', 'children'),
            Output('RV-selected-channels-psd', 'children'),
            Output('RV-flat-selected-channels-psd', 'children'),
            Output('RV-psd-graph', 'figure'),
        ],
        Input('RV-main-graph', 'selectedData'),
        [
            State('RV-raw', 'data'), State('RV-plotting-data', 'data'),
            State('RV-low-pass-input', 'value'),
            State('RV-notch-freq-input', 'value')
        ],
        prevent_initial_call=True
    )
    def get_selected_psd(selected_data, raw, plotting_data, low_pass, notch_freq):
        """Generates welch power-spectrum of selected data.
        Every trace with at least one datapoint within the selected segment is included.
        Flat channels are ignored and listed separately.
        """
        if not selected_data or (not selected_data['points']):
            raise PreventUpdate

        selected_interval_x = selected_data['range']['x']
        if selected_interval_x[0] < 0:
            selected_interval_x[0] = 0
        if selected_interval_x[1] > plotting_data['recording_length']:
            selected_interval_x[1] = plotting_data['recording_length']

        selected_interval_time = selected_interval_x[1] - selected_interval_x[0]
        selected_interval_text = f'{round(selected_interval_x[0], 1)} - {round(selected_interval_x[1], 1)} seconds'
        print(f'Selected interval: {selected_interval_text}')

        # Pick all channels that had at least one datapoint within the selected segment
        selected_channel_indices = [selected_point['curveNumber'] for selected_point in selected_data['points']]
        selected_channel_indices = list(set(selected_channel_indices))
        selected_channels = [plotting_data['selected_channels'][index] for index in selected_channel_indices if index < len(plotting_data['selected_channels'])]  # don't include model traces
        print('Selected channels: {}'.format(selected_channels))

        index_0 = raw.time_as_index(selected_interval_x[0])[0]
        index_1 = raw.time_as_index(selected_interval_x[1])[0]
        # print(index_0, index_1)

        data_subset, _ = raw[selected_channels, index_0:index_1]
        # print(data_subset.shape)

        # Ignore flat channels for psd and list them separately
        std = np.std(data_subset, 1)
        flat_channel_indices = np.where(std == 0)[0]
        flat_channel_names = []
        for i in flat_channel_indices:
            flat_channel_names.append(selected_channels.pop(i))
        if len(flat_channel_names) < 1:
            flat_channel_names = '-'
        print('Flat channels: {}'.format(flat_channel_names))

        # Psd parameters
        fs = raw.info['sfreq']
        n_fft_seconds = selected_interval_time if selected_interval_time <= 10 else 10
        n_fft = int(fs * n_fft_seconds)
        n_overlap = int(n_fft / 2)

        # Cut off psd slightly above notch frequency or low-pass frequency
        if notch_freq and notch_freq > low_pass:
            fmax = notch_freq + (notch_freq * 0.2)
        else:
            fmax = low_pass + (low_pass * 0.2)
        
        num_cores = multiprocessing.cpu_count()
        spectrum = raw.compute_psd('welch',
                                   picks=selected_channels,
                                   fmax=fmax,
                                   tmin=selected_interval_x[0],
                                   tmax=selected_interval_x[1],
                                   n_fft=n_fft,
                                   n_jobs=num_cores,
                                   n_overlap=n_overlap)

        freqs = spectrum.freqs[:]
        freqs = np.round(freqs, 2)

        # Scale power to V**2/Hz (taken from mne)
        Pxx_den = spectrum[:]
        Pxx_den *= (c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS ** 2)
        Pxx_den = np.log10(np.maximum(Pxx_den, np.finfo(float).tiny))
        Pxx_den *= 10
        Pxx_den = np.round(Pxx_den, 2)

        fig = get_power_spectrum_plot(freqs, Pxx_den, selected_channels, mean_Pxx_den=True)

        return selected_interval_text, ', '.join(selected_channels), ', '.join(flat_channel_names), fig
