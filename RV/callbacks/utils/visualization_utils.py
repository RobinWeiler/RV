import numpy as np

import RV.constants as c


def get_y_axis_ticks(selected_channels: list, channel_offset=c.DEFAULT_Y_AXIS_OFFSET, reorder_channels=False):
    """Generates ticks for y-axis along which to align EEG traces.

    Args:
        selected_channels (list): List of strings of plotted channel names.
        channel_offset (float, optional): Amount of space (in μV) between traces. Defaults to c.DEFAULT_Y_AXIS_OFFSET.
        reorder_channels (bool, optional): Whether or not to re-order traces according to brain lobes (only supported for EGI-129 montage). Defaults to False.

    Returns:
        np.array: Array of floats of y-axis ticks.
    """
    y_axis_ticks = np.arange(len(selected_channels))
    y_axis_ticks *= (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)

    if reorder_channels and all(channel_name in c.EGI129_CHANNELS for channel_name in selected_channels):
        lobe_offset = np.zeros_like(y_axis_ticks)

        lobe_names = list(c.CHANNELS_TO_LOBES_EGI129.keys())
        # Lobes are arranged top to bottom
        lobe_names.reverse()

        counter = 0
        for index, lobe in enumerate(lobe_names):
            for channel_name in c.CHANNELS_TO_LOBES_EGI129[lobe]:
                if channel_name in selected_channels:
                    lobe_offset[counter] = 2 * index * (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)
                    counter += 1

        y_axis_ticks += lobe_offset

    # Channels are arranged top to bottom
    y_axis_ticks = np.flip(y_axis_ticks)

    return y_axis_ticks
