import plotly.graph_objs as go

import numpy as np

import RV.constants as c


def get_power_spectrum_plot(freqs: np.array, all_Pxx_den: np.array, channel_names: list, mean_Pxx_den=True):
    """Generates power spectrum plot of given data.

    Args:
        freqs (np.array): Frequencies.
        all_Pxx_den (np.array): Power densities (in V**2/Hz).
        channel_names (list): List of strings of channels names corresponding to given power densities.
        mean_Pxx_den (bool, optional): Whether or not to include a mean power density. Defaults to True.

    Returns:
        plotly.graph_objs.Figure: Plot of power spectrum.
    """
    fig = go.Figure()

    if mean_Pxx_den:
        mean_Pxx_den = np.mean(all_Pxx_den, axis=0)
        fig.add_trace(
            go.Scattergl(
                x=freqs,
                y=mean_Pxx_den,
                name='Mean',
                hovertemplate='<b>%{fullData.name}</b> | Frequency = %{x:.1f} Hz, Power density = %{y:.1f} V**2/Hz' + '<extra></extra>',
                marker=dict(color='black')
            )
        )

    for i, Pxx_den in enumerate(all_Pxx_den):
        fig.add_trace(
            go.Scattergl(
                x=freqs,
                y=Pxx_den,
                name=channel_names[i],
                hovertemplate='<b>%{fullData.name}</b> | Frequency = %{x:.1f} Hz, Power density = %{y:.1f} V**2/Hz' + '<extra></extra>',
                opacity=0.6
            )
        )

    fig.update_layout(
        plot_bgcolor=c.PLOT_COLOR,
        paper_bgcolor=c.PLOT_COLOR,

        autosize=False,
        margin=dict(
            autoexpand=True,
            # l=0,
            # r=20,
            # b=0,
            t=20,
            pad=5,
        ),

        xaxis=dict(
            title_text='Frequencies (in Hz)',
        ),
        yaxis=dict(
            title_text='Power spectral density (in V**2/Hz)'
        ),
    )

    return fig
