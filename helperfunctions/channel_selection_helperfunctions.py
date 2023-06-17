from plotly.graph_objs import Figure, Scattergl

import numpy as np

from mne.viz.topomap import _get_pos_outlines

def get_channel_locations_plot(raw):
    """Generates channel-topography plot of given raw object.

    Args:
        raw (mne.io.Raw): Raw object to get channel locations from.

    Returns:
        plotly.graph_objs.Figure: Plot of channel-topography.
    """
    # Get channel locations through MNE
    if raw.info['dig']:
        pos, outlines = _get_pos_outlines(raw.info, np.arange(len(raw.info['chs'])), 'auto')
        
        head_coordinates = outlines['mask_pos']
        head_markers = []
        head_markers.append(min(head_coordinates[0]))
        head_markers.append(max(head_coordinates[0]))
        head_markers.append(min(head_coordinates[1]))
        head_markers.append(max(head_coordinates[1]))
        # print(head_markers)
    else:
        pos = np.array([])
        outlines = []
        head_markers = []
    # print(pos)
    # print(outlines)
    
    chs = raw.info['chs']
    channel_coordinates = pos  # np.empty((len(chs), 2))  # manual
    channel_names = []

    for index, channel in enumerate(chs):
        # channel_coordinates[index] = channel['loc'][:2]  # manual
        channel_names.append(channel['ch_name'])
        
    bad_channels = raw.info['bads']

    # Optional to scale channel locations
    # channel_coordinates = channel_coordinates * 1000

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
        
        if head_markers:
            topography_plot.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=head_markers[0],
                x1=head_markers[1],
                y0=head_markers[2],
                y1=head_markers[3],
                line_color="black",
            )

    topography_plot.update_layout(
        dragmode='select',
        showlegend=False,
        clickmode='event+select',
        # plot_bgcolor='#dfdfdf',
    )
    
    topography_plot.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=False,
    )
    
    topography_plot.update_xaxes(
        showgrid=False,
    )

    # topography_plot.update_xaxes(
    #     # title_text='Time (in seconds)'
    #     # showgrid=True,
    #     # zeroline=False,
    #     # constrain='domain',
    #     # range=(-0.2, 10.2),  # Start x-axis range to show approx. 10 seconds
    # )

    return topography_plot
