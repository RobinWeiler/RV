from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback, ctx
from dash.exceptions import PreventUpdate

import numpy as np

import RV.constants as c
from RV.callbacks.utils.visualization_utils import get_y_axis_ticks


def register_offset_callbacks():
    # Update offset using arrow-up and arrow-down keys on keyboard
    clientside_callback(
        """
            function(id) {
                document.addEventListener("keydown", function(event) {
                    if (event.target.nodeName != 'INPUT') {
                        if (event.key == 'ArrowUp') {
                            document.getElementById('RV-increase-offset-button').click()
                            event.stopPropogation()
                        }
                        if (event.key == 'ArrowDown') {
                            document.getElementById('RV-decrease-offset-button').click()
                            event.stopPropogation()
                        }
                    }
                });
                return window.dash_clientside.no_update       
            }
        """,
        Input('RV-offset-input', 'id')
    )

    @callback(
        Output('RV-offset-input', 'value', allow_duplicate=True),
        [
            Input('RV-increase-offset-button', 'n_clicks'),
            Input('RV-decrease-offset-button', 'n_clicks')
        ],
        State('RV-offset-input', 'value'),
        prevent_initial_call=True
    )
    def update_offset_input(increase_offset, decrease_offset, current_offset):
        """Updates RV-offset-input in steps of 10 based on clicks on hidden RV-increase-offset-button and RV-decrease-offset-button.
        The buttons are clicked when arrow-up and arrow-down keys on keyboard are pressed, respectively.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        if 'increase' in trigger:
            if current_offset != None:
                current_offset += 10
            else:
                current_offset = 50  # default 40 + 10 from event

        elif 'decrease' in trigger:
            if current_offset != None:
                if current_offset >= 10:
                    current_offset -= 10
                else:
                    # 0 as lower bound
                    current_offset = 0
            else:
                current_offset = 30  # default 40 - 10 from event

        return current_offset

    @callback(
        Output('RV-decrease-offset-button', 'disabled'),
        Input('RV-offset-input', 'value')
    )
    def disable_decrease_offset_button(current_offset):
        """Disables hidden RV-decrease-offset-button when current_offset is less than 10.
        """
        if current_offset is None:
            return False
        elif current_offset >= 10:
            return False
        else:
            return True

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True),
            Output('RV-main-graph-resampler', 'data', allow_duplicate=True),
        ],
        Input('RV-offset-input', 'value'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'), State('RV-model-data', 'data'),
            State('RV-main-graph', 'figure'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-scale-input', 'value'),
            State('RV-reorder-channels', 'value'),
        ],
        prevent_initial_call=True
    )
    def update_offset_graph(channel_offset, raw, plotting_data, model_data, current_fig, resampler, scale, reorder_channels):
        """Updates RV-main-graph (and resampler) when RV-offset-input changed.
        """
        if resampler is None:
            raise PreventUpdate

        print(f'New channel offset: {channel_offset}')

        data_subset, _ = raw[plotting_data['selected_channels'], :]
        if scale:
            data_subset *= (c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS * scale)
        else:
            data_subset *= c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS

        # Regenerate y-axis ticks
        y_axis_ticks = get_y_axis_ticks(plotting_data['selected_channels'], channel_offset, reorder_channels)
        data_subset += y_axis_ticks.reshape(-1, 1)

        # Update data in resampler
        for channel_index in range(len(plotting_data['selected_channels'])):
            resampler.hf_data[channel_index]['y'] = data_subset[channel_index]

        model_counter = 0
        for model in model_data.values():
            model_counter += 1
            y_axis_ticks = np.append(y_axis_ticks, (-2 * c.DEFAULT_Y_AXIS_OFFSET * model_counter))
            resampler.hf_data[len(plotting_data['selected_channels']) - 1 + model_counter]['y'] = model['predictions'] + y_axis_ticks[-1]

        # Trick to resample current view after data was updated
        relayout_data = {'xaxis.range[0]': current_fig['layout']['xaxis']['range'][0], 'xaxis.range[1]': current_fig['layout']['xaxis']['range'][1]}
        patched_fig = resampler.construct_update_data_patch(relayout_data)
        patched_fig['layout']['xaxis']['range'] = current_fig['layout']['xaxis']['range']

        patched_fig['layout']['yaxis']['tickvals'] = y_axis_ticks
        # if channel_offset == 0, put collapsed traces in the middle of the view, else spread from bottom to top
        patched_fig['layout']['yaxis']['range'][0] = (np.min(y_axis_ticks) - (2 * c.DEFAULT_Y_AXIS_OFFSET)) if channel_offset != 0 else (-(len(plotting_data['selected_channels']) + 2) * c.DEFAULT_Y_AXIS_OFFSET)

        return patched_fig, Serverside(resampler)
