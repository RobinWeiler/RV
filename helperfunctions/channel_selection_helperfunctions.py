from plotly.graph_objs import Figure, Scattergl

import numpy as np

from mne.io.pick import _pick_data_channels, pick_info
from mne.viz.topomap import _make_head_outlines, _find_topomap_coords
from mne.utils.check import _check_sphere


def get_channel_locations_plot(raw):
    """Generates channel-topography plot of given raw object.

    Args:
        raw (mne.io.Raw): Raw object to get channel locations from.

    Returns:
        plotly.graph_objs.Figure: Plot of channel-topography.
    """
    # Get channel locations through MNE
    if raw.info['dig']:
        channel_coords = _find_topomap_coords(raw.info, None)
        center = np.array([0, 0])  # np.mean(channel_coords, axis=0)
        distances = np.linalg.norm(channel_coords - center, axis=1)
        radius = np.max(distances)
        radius += (radius / 10)  # to put a bit of space between border and outer channels
        sphere = radius
        # print(sphere)

        sphere = _check_sphere(sphere, raw.info)
        # print(sphere)

        # picks = _pick_data_channels(raw.info, exclude=())  # pick only data channels
        # pos = pick_info(raw.info, picks)
        pos = _find_topomap_coords(raw.info, picks=None, sphere=sphere)
        pos = pos[:, :2]
        # print(pos)

        outlines_ = _make_head_outlines(sphere, pos, "head", (0.0, 0.0))
        outlines = {k: v for k, v in outlines_.items() if k not in ["patch"]}

    else:
        pos = np.array([])
        outlines = None
    
    chs = raw.info['chs']
    channel_coordinates = pos  # np.empty((len(chs), 2))  # manual
    channel_names = []

    for index, channel in enumerate(chs):
        channel_names.append(channel['ch_name'])
        
    bad_channels = raw.info['bads']

    topography_plot = Figure()

    if channel_coordinates.size > 0:
        for channel_index, channel in enumerate(channel_names):
            topography_plot.add_trace(
                Scattergl(
                    x=[channel_coordinates[channel_index, 0]],
                    y=[channel_coordinates[channel_index, 1]],
                    customdata=[channel],
                    mode="markers+text",
                    name=channel,
                    text=channel,
                    textposition="bottom center" if channel_coordinates[channel_index, 1] <= 0 else 'top center',
                    hovertemplate='<b>%{fullData.name}</b>' + '<extra></extra>' if channel not in bad_channels else '<b> Bad channel | %{fullData.name}</b>' + '<extra></extra>',
                    marker=dict(color='black') if channel not in bad_channels else dict(color='red')
                )
            )
        
        if outlines != None:
            for key, (x_coord, y_coord) in outlines.items():
                if "mask" in key or key in ("clip_radius", "clip_origin"):
                    continue

                outline = f"M {x_coord[0]},{y_coord[0]}"
                for xc, yc in zip(x_coord[1:], y_coord[1:]):
                    outline += f" L{xc},{yc}"

                topography_plot.add_shape(
                    dict(
                        type="path",
                        path=outline,
                        line_color='black',
                        line_width=2
                    )
                )

    topography_plot.update_layout(
        dragmode='select',
        showlegend=False,
        clickmode='event+select',
        plot_bgcolor='#fafafa',
        margin=dict(
            autoexpand=False,
            l=0,
            r=0,
            b=0,
            t=0,
            pad=5,
        ),
    )
    
    topography_plot.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=False,
        showticklabels=False
    )
    
    topography_plot.update_xaxes(
        showgrid=False,
        showticklabels=False
    )

    return topography_plot
