from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback, ctx
from dash.exceptions import PreventUpdate

import numpy as np

import RV.constants as c


def register_scale_callbacks():
    # Update scale using +/- keys on keyboard
    clientside_callback(
        """
            function(id) {
                document.addEventListener("keydown", function(event) {
                    if (event.target.nodeName != 'INPUT') {
                        if (event.key == '+' || event.key == '=') {
                            document.getElementById('RV-increase-scale-button').click()
                            event.stopPropogation()
                        }
                        if (event.key == '-' || event.key == '_') {
                            document.getElementById('RV-decrease-scale-button').click()
                            event.stopPropogation()
                        }
                    }
                });
                return window.dash_clientside.no_update       
            }
        """,
        Input('RV-scale-input', 'id')
    )

    @callback(
        Output('RV-scale-input', 'value', allow_duplicate=True),
        [
            Input('RV-increase-scale-button', 'n_clicks'),
            Input('RV-decrease-scale-button', 'n_clicks')
        ],
        State('RV-scale-input', 'value'),
        prevent_initial_call=True
    )
    def update_scale_input(increase_scale, decrease_scale, current_scale):
        """Updates RV-scale-input in steps of 0.5 based on clicks on hidden RV-increase-scale-button and RV-decrease-scale-button.
        The buttons are clicked when + and - keys on keyboard are pressed, respectively.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        if 'increase' in trigger:
            if current_scale != None:
                current_scale += 0.5
            else:
                current_scale = 1.5

        elif 'decrease' in trigger:
            if current_scale != None and current_scale >= 0.6:
                current_scale -= 0.5
            else:
                # 0.5 as lower bound
                current_scale = 0.5

        return current_scale

    @callback(
        Output('RV-decrease-scale-button', 'disabled'),
        Input('RV-scale-input', 'value')
    )
    def disable_decrease_scale_button(current_scale):
        """Disables hidden RV-decrease-scale-button when current_scale is less than 0.6.
        """
        if current_scale is None:
            return False
        elif current_scale >= 0.6:
            return False
        else:
            return True

    @callback(
        [
            Output('RV-main-graph', 'figure', allow_duplicate=True),
            Output('RV-main-graph-resampler', 'data', allow_duplicate=True),
        ],
        Input('RV-scale-input', 'value'),
        [
            State('RV-raw', 'data'), State('RV-plotting-data', 'data'),
            State('RV-main-graph', 'figure'),
            State('RV-main-graph-resampler', 'data'),
        ],
        prevent_initial_call=True
    )
    def update_scale_graph(scale, raw, plotting_data, current_fig, resampler):
        """Updates RV-main-graph (and resampler) when RV-scale-input changed.
        """
        if resampler is None:
            raise PreventUpdate

        print(f'New scale: {scale}')

        data_subset, _ = raw[plotting_data['selected_channels'], :]
        if scale:
            data_subset *= (c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS * scale)
        else:
            data_subset *= c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS
        data_subset += np.array(current_fig['layout']['yaxis']['tickvals'][:len(plotting_data['selected_channels'])]).reshape(-1, 1)

        # Update data in resampler
        for channel_index in range(len(plotting_data['selected_channels'])):
            resampler.hf_data[channel_index]['y'] = data_subset[channel_index]

        # Trick to resample current view after data was updated
        relayout_data = {'xaxis.range[0]': current_fig['layout']['xaxis']['range'][0], 'xaxis.range[1]': current_fig['layout']['xaxis']['range'][1]}
        patched_fig = resampler.construct_update_data_patch(relayout_data)
        patched_fig['layout']['xaxis']['range'] = current_fig['layout']['xaxis']['range']

        return patched_fig, Serverside(resampler)
