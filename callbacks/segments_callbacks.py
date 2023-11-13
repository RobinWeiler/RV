import math

import dash
from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

from helperfunctions.visualization_helperfunctions import get_EEG_plot

import globals

def register_segments_callbacks(app):

    # Enable/disable arrow buttons at edges
    @app.callback(
        [Output('left-button', 'disabled'), Output('right-button', 'disabled')],
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('show-annotations-only', 'value')]
        # prevent_initial_call=True
    )
    def _update_arrow_buttons(fig, segment_size, show_annotations_only):
        """Disables/enables arrow-buttons based on position of current segment. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, bool): Whether or not to disable left-arrow button, whether or not to disable right-arrow button.
        """
        left_disabled = True
        right_disabled = True

        if globals.plotting_data:
            if show_annotations_only:
                if globals.current_plot_index > 0:
                    left_disabled = False
                if globals.current_plot_index + 1 < len(globals.marked_annotations):
                    right_disabled = False
            elif segment_size:
                if globals.x0 == -0.5 and not globals.x1 > globals.plotting_data['EEG']['recording_length']:
                    right_disabled = False
                elif globals.x1 > globals.plotting_data['EEG']['recording_length']:
                    left_disabled = False
                else:
                    left_disabled = False
                    right_disabled = False

        return left_disabled, right_disabled

    # Enable/disable plus-/minus-10-seconds buttons at edges
    @app.callback(
        [Output('minus-ten-seconds-button', 'disabled'), Output('plus-ten-seconds-button', 'disabled')],
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('show-annotations-only', 'value'), State('EEG-graph', 'figure')]
        # prevent_initial_call=True
    )
    def _update_10_seconds_buttons(fig, segment_size, show_annotations_only, current_fig):
        """Disables/enables plus-/minus-10-seconds buttons based on position of current segment. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(bool, bool): Whether or not to disable minus-10-seconds button, whether or not to disable plus-10-seconds button.
        """

        left_disabled = True
        right_disabled = True

        if not segment_size:
            segment_size = 100  # hack for if statement below when recording is not segmented

        if globals.plotting_data:
            if show_annotations_only or segment_size <= 10:
                left_disabled = True
                right_disabled = True
            else:
                left_disabled = current_fig['layout']['xaxis']['range'][0] < globals.x0
                right_disabled = current_fig['layout']['xaxis']['range'][1] > globals.x1

        return left_disabled, right_disabled

    # Enable/disable segment-slider
    @app.callback(
        [Output('segment-slider', 'disabled'), Output('segment-slider', 'max'), Output('segment-slider', 'step'), Output('segment-slider', 'marks')],
        [Input('EEG-graph', 'figure'), Input('segment-size', 'value'), Input('show-annotations-only', 'value')],
        State('segment-slider', 'disabled')
        # prevent_initial_call=True
    )
    def _update_segment_slider(fig, segment_size, show_annotations_only, segment_slider_disabled):
        """Disables/enables segment-slider. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, int, int, dict): Whether or not to disable segment-slider, max value, step size, dict of marks.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if globals.plotting_data and segment_size:
            if 'EEG-graph' in trigger and not segment_slider_disabled:
                raise PreventUpdate
            if show_annotations_only and len(globals.marked_annotations) > 0:
                num_segments = int(len(globals.marked_annotations) - 1)
                marks = {i: '{}'.format(i) for i in range(num_segments + 1)}
            else:
                num_segments = int(globals.plotting_data['EEG']['recording_length'] // segment_size)
                marks = {i: '{} - {}'.format(i * segment_size, i * segment_size + segment_size) for i in range(num_segments + 1)}

            return False, num_segments, 1, marks
        else:
            return True, 1, 1, {0: '0', 1: '1'}

    # Switch plotted segment via segment-slider or arrow buttons
    @app.callback(
        [Output('EEG-graph', 'figure', allow_duplicate=True), Output('segment-slider', 'value')],
        [Input('segment-slider', 'value'), Input('left-button', 'n_clicks'), Input('right-button', 'n_clicks')],
        [
            State('segment-size', 'value'), State('show-annotations-only', 'value'), 
            State('use-slider', 'value'), State('skip-hoverinfo', 'value'), State('hide-bad-channels-button', 'n_clicks'), State('highlight-model-channels-button', 'n_clicks'),
            State('annotation-label', 'value'), State('EEG-graph', 'figure')
        ],
        prevent_initial_call=True
    )
    def _use_segment_slider(segment_slider, left_button, right_button, segment_size, show_annotations_only, use_slider, skip_hoverinfo, hide_bad_channels, highlight_model_channels, annotation_label, current_fig):
        """Moves viewed segment. Triggered when segment-slider is moved and when left- or right-arrow button is clicked.

        Args:
            segment_slider (int): Current value of segment-sldier.
            left_button (int): Num clicks on left-arrow button.
            right_button (int): Num clicks on right-arrow button.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.
            use_slider (bool): Whether or not to activate view-slider.
            skip_hoverinfo (bool): Whether or not to activate hover-info.
            hide_bad_channels (int): Num clicks on hide-bad-channels-button button.
            highlight_model_channels (int): Num clicks on highlight-model-channels-button button.
            annotation_label (string); Label for new annotations.
            current_fig (plotly.graph_objs.Figure): The current EEG plot.

        Returns:
            tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if globals.plotting_data and segment_size:
            # Switching to next segment, current segment gets updated in globals.preloaded_plots
            if not show_annotations_only:
                globals.preloaded_plots[globals.current_plot_index] = current_fig

            if 'segment-slider' in trigger:
                globals.current_plot_index = segment_slider
            elif 'left-button' in trigger:
                globals.current_plot_index -= 1
            elif 'right-button' in trigger:
                globals.current_plot_index += 1

            if show_annotations_only and (len(globals.marked_annotations) > globals.current_plot_index):
                globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2

            elif 'segment-slider' in trigger:
                globals.x0 = segment_size * segment_slider - 0.5 if segment_slider != 0 else -0.5
                globals.x1 = segment_size + (segment_size * segment_slider) + 0.5
            elif 'left-button' in trigger:
                globals.x0 -= segment_size
                globals.x1 -= segment_size

            elif 'right-button' in trigger:
                globals.x0 += segment_size
                globals.x1 += segment_size

            updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

            return updated_fig, globals.current_plot_index
        else:
            raise PreventUpdate

    # Move view by 10 seconds left or right
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        [Input('minus-ten-seconds-button', 'n_clicks'), Input('plus-ten-seconds-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def _use_10_seconds_buttons(minus_10_seconds, plus_10_seconds):
        """Moves viewed segment. Triggered when minus-10-seconds or plus-10-seconds button is clicked.

        Args:
            minus_10_seconds (int): Num clicks on minus-10-seconds button.
            plus_10_seconds (int): Num clicks on plus-10-seconds button.

        Returns:
            tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # print(trigger)
        patched_fig = Patch()

        if globals.plotting_data:
            if 'minus' in trigger:
                patched_fig['layout']['xaxis']['range'][0] -= 10
                patched_fig['layout']['xaxis']['range'][1] -= 10
            elif 'plus' in trigger:
                patched_fig['layout']['xaxis']['range'][0] += 10
                patched_fig['layout']['xaxis']['range'][1] += 10

            return patched_fig
        else:
            raise PreventUpdate

    # Update plot when segment_size is changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('segment-size', 'value'),
        [
            State('show-annotations-only', 'value'), State('use-slider', 'value'), 
            State('skip-hoverinfo', 'value'), State('hide-bad-channels-button', 'n_clicks'), State('highlight-model-channels-button', 'n_clicks'),
            State('annotation-label', 'value')
        ],
        prevent_initial_call=True
    )
    def _use_segment_slider(segment_size, show_annotations_only, use_slider, skip_hoverinfo, hide_bad_channels, highlight_model_channels, annotation_label):
        """Moves viewed segment. Triggered when segment-slider is moved and when left- or right-arrow button is clicked.

        Args:
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.
            use_slider (bool): Whether or not to activate view-slider.
            skip_hoverinfo (bool): Whether or not to activate hover-info.
            hide_bad_channels (int): Num clicks on hide-bad-channels-button button.
            highlight_model_channels (int): Num clicks on highlight-model-channels-button button.
            annotation_label (string); Label for new annotations.

        Returns:
            tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
        """
        if globals.plotting_data:
            if segment_size:
                globals.x1 = globals.x0 + segment_size + 1
            else:
                globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

            updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only, skip_hoverinfo, (hide_bad_channels % 2 != 0), (highlight_model_channels % 2 != 0))

            return updated_fig

        else:
            raise PreventUpdate

    # @app.callback(
    #     Output('preload-data', 'children'),
    #     [Input('EEG-graph', 'figure'), Input('bad-channels-dropdown', 'value'), Input('segment-size', 'value'), Input('show-annotations-only', 'value'), Input('annotation-label-color', 'value')],
    #     [State('use-slider', 'value'), State('annotation-label', 'value')],
    #     prevent_initial_call=True
    # )
    # def _preload_plots(current_fig, current_bad_channels, segment_size, show_annotations_only, annotation_color, use_slider, annotation_label):
    #     """Preloads 1 following segment and adds it to globals.preloaded_plots. Triggered when EEG plot has loaded.

    #     Args:
    #         fig (plotly.graph_objs.Figure): EEG plot.
    #         segment_size (int): Segment size of EEG plot.
    #         use_slider (bool): Whether or not to activate view-slider.
    #     """
    #     trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
    #     # print(trigger)

    #     if globals.plotting_data:
    #         if segment_size:
    #             num_segments = math.ceil(globals.plotting_data['EEG']['recording_length'] / segment_size)
    #             # print(num_segments)
                
    #             # upper_bound = globals.current_plot_index + 2 if globals.current_plot_index + 2 < num_segments else num_segments
    #             # # print(upper_bound)

    #             # globals.preloaded_plots[globals.current_plot_index] = fig

    #             # for segment_index in range(upper_bound):
    #                 # if segment_index not in globals.preloaded_plots:
    #                 #     new_x0 = segment_index * segment_size - 0.5
    #                 #     new_x1 = segment_size + segment_index * segment_size + 0.5
    #                 #     globals.preloaded_plots[segment_index] = get_EEG_plot(globals.plotting_data, new_x0, new_x1, use_slider)
    #                 #     print(segment_index)

    #             preloaded_segments = list(globals.preloaded_plots.keys())
    #             for key in preloaded_segments:
    #                 if int(key) > globals.current_plot_index + 1 or int(key) < globals.current_plot_index - 1:
    #                     # print('Removing segment {} for preloaded plots'.format(key))
    #                     globals.preloaded_plots.pop(key)

    #             if ('bad-channels' in trigger) or ('segment-size' in trigger) or ('show-annotations-only' in trigger) or ('annotation-label-color' in trigger):
    #                 print('Deleting preloaded segments')
    #                 globals.preloaded_plots.clear()

    #             if (not show_annotations_only) and (globals.current_plot_index + 1 not in globals.preloaded_plots.keys()) and (globals.current_plot_index + 1 < num_segments):
    #                 print('Preloading segments')
    #                 new_x0 = globals.x0 + segment_size
    #                 new_x1 = globals.x1 + segment_size
    #                 globals.preloaded_plots[globals.current_plot_index + 1] = get_EEG_plot(globals.plotting_data, new_x0, new_x1, annotation_label, use_slider, show_annotations_only)
    #                 print('Next segment preloaded')

    #             print(globals.preloaded_plots.keys())
