import warnings

import numpy as np
from numpy import pi, sin, cos

import mne
from mne._fiff.pick import _pick_data_channels, pick_info
from mne.channels.layout import _pair_grad_sensors, _find_topomap_coords, _merge_ch_data
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _setup_interp
from mne.viz.utils import _setup_vmin_vmax
from mne.utils.check import _check_sphere

import plotly.graph_objs as go

import RV.constants as c


def degrees_to_radians(degrees: int):
    """Converts degrees to radians.

    Args:
        degrees (int): Angle.

    Returns:
        float: Radians.
    """
    return degrees * pi / 180

def get_clipping_svg_path(radius: int, position: str, center=(0,0), start_angle=180, end_angle=0, num_points=64):
    """Generates SVG paths for half-circles used to clip data outside of head outline.

    Args:
        radius (int): Radius of half-circle.
        position (str): Whether clipping is positioned on top or on bottom of head outline. Can be 'top' or 'bottom'.
        center (tuple, optional): Center of half-circle. Defaults to [0,0].
        start_angle (int, optional): Where to start drawing half-circle (in degrees). Defaults to 180.
        end_angle (int, optional): Where to end drawing half-circle (in degrees). Defaults to 0.
        num_points (int, optional): Number of points to draw half-circle. More points result in higher resolution Defaults to 64.

    Returns:
        str: SVG path.
    """
    start_radian = degrees_to_radians(start_angle)
    end_radian = degrees_to_radians(end_angle)

    t = np.linspace(start_radian, end_radian, num_points)
    x = center[0] + (radius * cos(t))
    y = center[1] + (radius * sin(t))

    path = f'M {x[1]},{y[1]}'

    for xc, yc in zip(x[2:], y[2:]):
        path += f' L{xc},{yc}'

    if position == 'top':
        path += f' L{radius},{radius}'
        path += f' L{-radius},{radius}'
    elif position == 'bottom':
        path += f' L{radius},{-radius}'
        path += f' L{-radius},{-radius}'

    path += ' Z'

    return path

def get_topomap(
    data: np.array,
    info: mne.Info,
    num_contours=6,
    res=64,
    vmin=None,
    vmax=None,
    cmap='RdBu_r',
    plot_colorbar=True,
    colorbar_title='μV',
):
    """Generate topomap of given data.

    Args:
        data (np.array): EEG data of one timepoint.
        info (mne.Info): Info object of given EEG data.
        num_contours (int, optional): Number of contours in topomap. Defaults to 6.
        res (int, optional): Resolution of interpolation. Defaults to 64.
        vmin (float, optional): Minimum value of colorscale. Defaults to None.
        vmax (float, optional): Maximum value of colorscale. Defaults to None.
        cmap (str, optional): Colorscale. Defaults to 'RdBu_r'.
        plot_colorbar (bool, optional): Whether or not to plot colorbar. Defaults to True.
        colorbar_title (str, optional): Title for colorbar. Defaults to 'μV'.

    Returns:
        plotly.graph_objects.Figure: Topomap plot.
    """
    # Check if number of channels in info matches data
    if len(info['chs']) != data.shape[0]:
        raise ValueError(f"Number of channels in the Info object ({len(info['chs'])}) and the data array ({data.shape[0]}) do not match.")

    if data.ndim > 1:
        raise ValueError(f'Data needs to be array of shape (n_sensors,); got shape {data.shape}.')

    fig = go.Figure()

    channel_coords = _find_topomap_coords(info, None)

    channel_names = info['ch_names']

    if 'Cz' in info.ch_names:
        cz_coord = _find_topomap_coords(info, 'Cz')[0]

        # Move all channels by the offset Cz has to the center (0,0)
        channel_coords = channel_coords - cz_coord

    distances = np.linalg.norm(channel_coords, axis=1)

    # Set radius to max distance of channels to center
    radius = np.max(distances)
    radius += (radius / 20)  # put a bit of space between border and outer channels

    sphere = np.array([0, 0, 0, radius])

    outlines = _make_head_outlines(sphere, channel_coords, 'head', (0.0, 0.0))
    assert isinstance(outlines, dict)

    extent, Xi, Yi, interp = _setup_interp(channel_coords, res, 'cubic', 'head', outlines, 'mean')
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # Heatmap-like plot
    fig.add_trace(go.Contour(
        autocontour=True,
        colorscale=cmap,
        contours_coloring='heatmap',
        hoverinfo='none',
        line=dict(
            width=0,  # no contour lines
        ),
        ncontours=1,
        showscale=False,
        x=Xi[0],
        y=[y[0] for y in Yi],
        z=Zi,
        zmax=vmax,
        zmin=vmin
    ))

    if num_contours != 0 and not ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        # Draw contour lines
        fig.add_trace(
            go.Contour(
                autocontour=True,
                contours_coloring='none',
                hoverinfo='none',
                line=dict(
                    width=1,
                    color='black'
                ),
                ncontours=(num_contours + 1),
                showscale=False,
                x=Xi[0],
                y=[y[0] for y in Yi],
                z=Zi
            )
        )

    # Clip heatmap outside of head circle
    top_clip = get_clipping_svg_path(radius=extent[1], position='top', start_angle=180, end_angle=0)
    fig.add_shape(
        dict(
            type='path',
            path=top_clip,
            fillcolor=c.PLOT_COLOR,
            line_color=c.PLOT_COLOR,
            line_width=1
        )
    )

    bottom_clip = get_clipping_svg_path(radius=extent[1], position='bottom', start_angle=180, end_angle=360)
    fig.add_shape(
        dict(
            type='path',
            path=bottom_clip,
            fillcolor=c.PLOT_COLOR,
            line_color=c.PLOT_COLOR,
            line_width=1
        )
    )

    # Plot channel coordinates
    for channel_index, channel in enumerate(channel_coords):
        marker_line_width = 0
        marker_size = 3
        marker_color = 'black'

        fig.add_trace(
            go.Scatter(
                hoverinfo='all',
                hovertemplate='<b>%{fullData.name}</b><extra></extra>',
                marker=dict(
                    color=marker_color,
                    size=marker_size,
                    line=dict(width=marker_line_width),
                ),
                mode='markers',
                name=channel_names[channel_index],
                text=channel_names[channel_index],
                textposition='bottom center' if channel[1] <= 0 else 'top center',
                x=[channel[0]],
                y=[channel[1]],
            )
        )

    # Draw head outline
    if isinstance(outlines, dict):
        outlines_ = {k: v for k, v in outlines.items() if k not in ['patch']}
        for key, (x_coord, y_coord) in outlines_.items():
            if 'mask' in key or key in ('clip_radius', 'clip_origin'):
                continue

            outline_svg_path = f'M {x_coord[0]},{y_coord[0]}'
            for xc, yc in zip(x_coord[1:], y_coord[1:]):
                outline_svg_path += f' L{xc},{yc}'

            fig.add_shape(
                dict(
                    type='path',
                    path=outline_svg_path,
                    line_color='black',
                    line_width=2
                )
            )

    # Plot colorbar
    if plot_colorbar:
        fig.add_trace(
            go.Scatter(
                hoverinfo='none',
                marker=dict(
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(
                        x=0.8,
                        thickness=20,
                        len=0.9,
                        tickvals=np.round(np.linspace(vmin, vmax, 6)),
                        title=dict(
                            text=colorbar_title,
                            font=dict(
                                # size=18,
                                color='black')
                        ),
                    ),
                    colorscale=cmap,
                    showscale=True
                ),
                mode='markers',
                x=[None],
                y=[None]
            )
        )

    fig.update_layout(
        autosize=False,
        margin=dict(
            autoexpand=False,
            # b=20,
            l=0,
            pad=5,
            r=0,
            t=0,
        ),
        modebar_remove=['select2d', 'lasso2d'],
        paper_bgcolor=c.PLOT_COLOR,
        plot_bgcolor=c.PLOT_COLOR,
        showlegend=False
    )

    fig.update_xaxes(
        showticklabels=False,
    )

    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1,
        showticklabels=False,
    )

    return fig, outlines, channel_coords

def get_topomap_heatmap(
    data: np.array,
    channel_coords : np.array,
    outlines: dict,
    res=64,
    vmin=None,
    vmax=None,
    cmap='RdBu_r',
):
    """Generates topomap-heatmap plot for animated frames.

    Args:
        data (np.array): EEG data of one timepoint.
        channel_coords (np.array): xy-coordinates of EEG channels.
        outlines (dict): Head outline.
        res (int, optional): Resolution of interpolation. Defaults to 64.
        vmin (float, optional): Minimum value of colorscale. Defaults to None.
        vmax (float, optional): Maximum value of colorscale. Defaults to None.
        cmap (str, optional): Colorscale. Defaults to 'RdBu_r'.

    Returns:
        plotly.graph_objects.Contour: Topomap-heatmap plot.
    """
    _, Xi, Yi, interp = _setup_interp(
        channel_coords, res, 'cubic', 'head', outlines, 'mean'
    )
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # Heatmap-like plot
    topoheatmap = go.Contour(
        autocontour=True,
        colorscale=cmap,
        contours_coloring='heatmap',
        hoverinfo='none',
        line=dict(
            width=0,  # no contour lines
        ),
        ncontours=1,
        showscale=False,
        x=Xi[0],
        y=[y[0] for y in Yi],
        z=Zi,
        zmax=vmax,
        zmin=vmin
    )

    return topoheatmap

def get_animated_topomap_fig(raw, x0, x1, interpolate_bads=True):
    """Generates animation of topomap from given mne.io.Raw object and time interval.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        x0 (float): Start of selected time interval.
        x1 (float): End of selected time interval.
        interpolate_bads (bool, optional): Whether or not to interpolate bad channels. Defaults to True.

    Returns:
        plotly.graph_objs.Figure: Animated topomap.
    """
    index_0 = raw.time_as_index(x0)[0]
    index_1 = raw.time_as_index(x1)[0]

    if interpolate_bads:
        raw = raw.interpolate_bads(reset_bads=True, mode='accurate', method='spline', verbose=0)

    topomap_data, _ = raw[:, index_0:index_1]
    topomap_data = topomap_data * c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS

    vmin = np.min(topomap_data)
    vmax = np.max(topomap_data)

    frames = []
    first_frame, outlines, channel_coords = get_topomap(topomap_data[:, 0], raw.info, num_contours=0, vmin=vmin, vmax=vmax, colorbar_title='μV')

    for timepoint_index in range(topomap_data.shape[1]):
        print('{} / {} frames'.format(timepoint_index, topomap_data.shape[1]))
        frames.append(
            go.Frame(
                data=get_topomap_heatmap(topomap_data[:, timepoint_index], channel_coords, outlines, vmin=vmin, vmax=vmax),
                name=str(raw.times[(index_0 + timepoint_index)])
            )
        )

    animated_power_topomap_fig = go.Figure(frames=frames)

    # Initialize initial frame of animation
    for i in range(len(first_frame.data)):
        animated_power_topomap_fig.add_trace(first_frame.data[i],)
    animated_power_topomap_fig.layout = first_frame.layout
    
    def frame_args(duration):
        return {
            'frame': {'duration': duration, 'redraw': True},
            'fromcurrent': True,
            'mode': 'immediate',
            'transition': {'duration': duration, 'easing': 'linear'},
        }

    slider = [
        {
            'len': 0.9,
            'pad': {'b': 10, 't': 60},
            'steps': [
                {
                    'args': [[f.name], frame_args(0)],
                    'label': f.name,
                    'method': 'animate',
                }
                for f in animated_power_topomap_fig.frames
            ],
            'x': 0.1,
            'y': 0.2,
        }
    ]

    animated_power_topomap_fig.update_layout(
        updatemenus = [
            {
                'buttons': [
                    {
                        'args': [None, frame_args(50)],
                        'label': '&#9654;', # play symbol
                        'method': 'animate',
                    },
                    {
                        'args': [[None], frame_args(0)],
                        'label': '&#10074; &#10074;', # pause symbol
                        'method': 'animate',
                    },
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 70},
                'type': 'buttons',
                'x': 0.1,
                'y': 0.2,
            }
        ],
        sliders=slider
    )

    return animated_power_topomap_fig
