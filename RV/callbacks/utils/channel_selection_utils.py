import plotly.graph_objs as go

from mne.channels.layout import _find_topomap_coords
from mne.utils.check import _check_sphere
from mne.viz.topomap import _make_head_outlines

import mne
import numpy as np

import RV.constants as c


def get_channel_topography_plot(raw: mne.io.Raw):
    """Generates channel-topography plot of given mne.io.Raw object.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.

    Returns:
        go.Figure: Plotly figure with channel-topography.
    """
    topography_plot = go.Figure()

    # If channel-location data is not available
    if not raw.info['dig']:
        channel_coords = np.array([])
        outlines = None

        return topography_plot

    # raw = raw.pick('data')
    channel_coords = _find_topomap_coords(raw.info, None)

    if 'Cz' in raw.info.ch_names:
        cz_coord = _find_topomap_coords(raw.info, 'Cz')[0]

        # Move all channels by the offset Cz has to the center (0,0)
        channel_coords = channel_coords - cz_coord

    distances = np.linalg.norm(channel_coords, axis=1)

    # Set radius to max distance of channels to center
    radius = np.max(distances)
    radius += (radius / 10)  # put a bit of space between border and outer channels

    sphere = radius
    sphere = _check_sphere(sphere, raw.info)

    outlines_temp = _make_head_outlines(sphere, channel_coords, 'head', (0.0, 0.0))
    outlines = {k: v for k, v in outlines_temp.items() if k not in ['patch']}

    bad_channels = raw.info['bads']

    # Plot channel locations
    for channel_index, channel in enumerate(raw.ch_names):
        topography_plot.add_trace(
            go.Scattergl(
                x=[channel_coords[channel_index, 0]],
                y=[channel_coords[channel_index, 1]],
                customdata=[channel],
                hovertemplate='<b>%{fullData.name}</b>' + '<extra></extra>' if channel not in bad_channels else '<b>%{fullData.name} (bad channel)</b>' + '<extra></extra>',
                marker=dict(color='black') if channel not in bad_channels else dict(color=c.BAD_CHANNEL_COLOR),
                mode='markers+text',
                name=channel,
                text=channel,
                textposition='bottom center' if channel_coords[channel_index, 1] <= 0 else 'top center'
            )
        )

    # Add head outlines to plot
    for key, (x_coord, y_coord) in outlines.items():
        if 'mask' in key or key in ('clip_radius', 'clip_origin'):
            continue

        outline = f'M {x_coord[0]},{y_coord[0]}'
        for xc, yc in zip(x_coord[1:], y_coord[1:]):
            outline += f' L{xc},{yc}'

        topography_plot.add_shape(
            dict(
                line_color='black',
                line_width=2,
                path=outline,
                type='path'
            )
        )

    topography_plot.update_layout(
        autosize=False,
        clickmode='event+select',
        dragmode='select',
        font=dict(size=12 if len(raw.ch_names) <= 64 else 8),
        margin=dict(
            autoexpand=False,
            l=0,
            r=0,
            b=0,
            t=0,
            pad=5,
        ),
        plot_bgcolor=c.PLOT_COLOR,
        showlegend=False,
    )
    
    topography_plot.update_yaxes(
        scaleanchor = 'x',
        scaleratio = 1,

        showgrid=False,
        showticklabels=False
    )

    topography_plot.update_xaxes(
        showgrid=False,
        showticklabels=False
    )

    return topography_plot

def get_10_20_channels(channel_names: list):
    """Picks 10-20 channels from given channel_names.

    Args:
        channel_names (list): List of strings of all available channel names.

    Raises:
        Exception: If 10-20 channels cannot be found.

    Returns:
        list: List of strings of 10-20-channel names.
    """
    if all(channel in channel_names for channel in c.STANDARD_10_20):
        selected_channels = c.STANDARD_10_20
    elif all(channel in channel_names for channel in c.BIOSEMI64_10_20):
        selected_channels = c.BIOSEMI64_10_20
    elif all(channel in channel_names for channel in c.TUAR_CHANNELS):
        selected_channels = c.TUAR_CHANNELS
    elif all(channel in channel_names for channel in c.EGI128_10_20):
        selected_channels = c.EGI128_10_20
    elif all(channel in channel_names for channel in c.EGI128_2_10_20):
        selected_channels = c.EGI128_2_10_20
    elif all(channel in channel_names for channel in c.EGI129_10_20):
        selected_channels = c.EGI129_10_20
    elif all(channel in channel_names for channel in c.ADJACENT_10_20):
        selected_channels = c.ADJACENT_10_20
    else:
        raise Exception(f'Could not find 10-20 channels in {channel_names}.')

    return selected_channels
