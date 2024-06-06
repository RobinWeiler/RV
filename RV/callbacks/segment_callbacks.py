from dash import Patch
from dash_extensions.enrich import Output, Input, State, callback, clientside_callback, no_update, ctx
from dash.exceptions import PreventUpdate

import numpy as np


def register_segment_callbacks():
    # Simulate click on left- and right-arrow buttons when left- and right-arrow keys are pressed on keyboard
    clientside_callback(
        """
            function(id) {
                document.addEventListener("keydown", function(event) {
                    if (event.target.nodeName != 'INPUT') {
                        if (event.key == 'ArrowLeft') {
                            document.getElementById('RV-left-arrow-button').click()
                            event.stopPropogation()
                        }
                        if (event.key == 'ArrowRight') {
                            document.getElementById('RV-right-arrow-button').click()
                            event.stopPropogation()
                        }
                    }
                });
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-segment-slider', 'id'),
        Input('RV-segment-slider', 'id')
    )

    @callback(
        Output('RV-segment-slider', 'value', allow_duplicate=True),
        [
            Input('RV-left-arrow-button', 'n_clicks'), Input('RV-right-arrow-button', 'n_clicks')
        ],
        State('RV-segment-slider', 'value'),
        prevent_initial_call=True
    )
    def use_arrow_buttons(left_button, right_button, current_segment):
        """Moves RV-segment-slider left/right respective to clicks on left- and right-arrow buttons.
        """
        trigger = ctx.triggered_id
        # print(trigger)

        if 'left' in trigger:
            current_segment -= 1

            return current_segment

        elif 'right' in trigger:
            current_segment += 1

            return current_segment

        raise PreventUpdate

    @callback(
        [
            Output('RV-left-arrow-button', 'disabled'), Output('RV-right-arrow-button', 'disabled')
        ],
        Input('RV-segment-slider', 'value'),
        [
            State('RV-segment-slider', 'max'),
            State('RV-left-arrow-button', 'disabled'), State('RV-right-arrow-button', 'disabled')
        ],
        prevent_initial_call=True
    )
    def disable_arrow_buttons(current_segment, num_segments, left_disabled, right_disabled):
        """Disables/enables arrow buttons based on value of RV-segment-slider.
        """
        if current_segment is None:
            return True, True

        if current_segment > 0:
            if left_disabled == False:
                left_disabled = no_update
            else:
                left_disabled = False
        else:
            left_disabled = True

        if current_segment < num_segments:
            if right_disabled == False:
                right_disabled = no_update
            else:
                right_disabled = False
        else:
            right_disabled = True

        return left_disabled, right_disabled

    @callback(
        Output('RV-main-graph', 'figure', allow_duplicate=True),
        Input('RV-segment-slider', 'value'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-segment-size-input', 'value'),
            State('RV-annotations-only-mode', 'value')
        ],
        prevent_initial_call=True,
    )
    def switch_segments(segment_slider, raw, plotting_data, resampler, segment_size, annotations_only_mode):
        """Switches plotted segment. Triggered by RV-segment-slider.
        """
        if resampler is None:
            raise PreventUpdate

        if annotations_only_mode:
            x_axis_range_0 = raw.annotations.onset[segment_slider] - 1
            x_axis_range_1 = raw.annotations.onset[segment_slider] + raw.annotations.duration[segment_slider] + 1
        else:
            if segment_size:
                x_axis_range_0 = (segment_slider * segment_size) - 0.5
                x_axis_range_1 = (segment_slider * segment_size) + segment_size + 0.5
            else:
                x_axis_range_0 = -0.5
                x_axis_range_1 = plotting_data['recording_length'] + 0.5

        # Update plotted view with resampler
        relayout_data = {'xaxis.range[0]': x_axis_range_0, 'xaxis.range[1]': x_axis_range_1}
        patched_fig = resampler.construct_update_data_patch(relayout_data)
        patched_fig['layout']['xaxis']['range'] = (x_axis_range_0, x_axis_range_1)

        return patched_fig

    @callback(
        Output('RV-segment-slider', 'value', allow_duplicate=True),
        Input('RV-annotation-overview-graph', 'clickData'),
        prevent_initial_call=True
    )
    def click_annotation_overview(click_data):
        """Set RV-segment-slider to corresponding segment of point clicked on in RV-annotation-overview-graph.
        """
        clicked_segment = click_data['points'][0]['pointNumber']

        return clicked_segment

    @callback(
        [
            Output('RV-segment-slider', 'max', allow_duplicate=True),
            Output('RV-segment-slider', 'marks', allow_duplicate=True),
            Output('RV-segment-slider', 'value', allow_duplicate=True),
            Output('RV-annotation-overview-graph', 'figure', allow_duplicate=True),
        ],
        Input('RV-segment-size-input', 'value'),
        [
            State('RV-plotting-data', 'data'),
            State('RV-segment-slider', 'value'),
            State('RV-annotations-only-mode', 'value'),
            State('RV-annotation-overview-graph', 'figure'),
        ],
        prevent_initial_call=True
    )
    def update_segment_slider(segment_size, plotting_data, segment_slider, annotations_only_mode, current_annotation_fig):
        """Changes RV-segment-slider parameters based on RV-segment-size-input.
        """
        if plotting_data is None or annotations_only_mode:
            # if annotations_only_mode is active, RV-segment-slider is changed by a callback in annotations_callbacks.py
            raise PreventUpdate

        # Calculate new RV-segment-slider parameters
        if segment_size:
            num_segments = int(plotting_data['recording_length'] // segment_size)
            segment_slider_marks = {i: {'label': f'{i * segment_size}'} for i in range(num_segments + 1)}
            segment_slider = 0
        else:
            num_segments = 0
            segment_slider_marks = {0: {'label': '0'}}
            segment_slider = 0

        # Update clickable points in RV-annotation-overview-graph with new RV-segment-slider parameters
        if segment_size:
            x_ticks = [(int(segment) * segment_size) for segment in segment_slider_marks.keys()]
            x_ticks = np.array(x_ticks)
            # x_ticks = [(segment + (segment_size / 2)) for segment in text_labels]

            current_annotation_fig['data'][0]['x'] = x_ticks
            current_annotation_fig['data'][0]['y'] = np.repeat(0.9, len(x_ticks))
            current_annotation_fig['data'][0]['text'] = x_ticks
        else:
            current_annotation_fig['data'][0]['x'] = None
            current_annotation_fig['data'][0]['y'] = None
            current_annotation_fig['data'][0]['text'] = None

        return num_segments, segment_slider_marks, segment_slider, current_annotation_fig
