import dash
from dash.dependencies import Input, Output, State

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

    # Enable/disable segment-slider
    @app.callback(
        [Output('segment-slider', 'disabled'), Output('segment-slider', 'max'), Output('segment-slider', 'step'), Output('segment-slider', 'marks')],
        Input('EEG-graph', 'figure'),
        [State('segment-size', 'value'), State('show-annotations-only', 'value')]
        # prevent_initial_call=True
    )
    def _update_segment_slider(fig, segment_size, show_annotations_only):
        """Disables/enables segment-slider. Triggered when EEG plot has loaded.

        Args:
            fig (plotly.graph_objs.Figure): EEG plot.
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(bool, bool): Whether or not to disable segment-slider, max value.
        """
        if globals.plotting_data and segment_size:
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
        [State('segment-size', 'value'), State('show-annotations-only', 'value'), State('use-slider', 'value'), State('annotation-label', 'value'), State('EEG-graph', 'figure')],
        prevent_initial_call=True
    )
    def _use_segment_slider(segment_slider, left_button, right_button, segment_size, show_annotations_only, use_slider, annotation_label, current_fig):
        """Moves viewed segment. Triggered when segment-slider is moved and when left- or right-arrow button is clicked.

        Args:
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
        """
        trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(trigger)

        if globals.plotting_data and segment_size:
            if 'segment-slider' in trigger:
                globals.current_plot_index = segment_slider
            elif 'left-button' in trigger:
                globals.current_plot_index -= 1
            elif 'right-button' in trigger:
                globals.current_plot_index += 1

            if show_annotations_only and len(globals.marked_annotations) > globals.current_plot_index:
                globals.x0 = globals.marked_annotations[globals.current_plot_index][0] - 2
                globals.x1 = globals.marked_annotations[globals.current_plot_index][1] + 2

            elif 'segment-slider' in trigger:
                globals.x0 = segment_size * segment_slider if segment_slider != 0 else -0.5
                globals.x1 = segment_size + (segment_size * segment_slider) + 0.5
            elif 'left-button' in trigger:
                globals.x0 -= segment_size
                globals.x1 -= segment_size

            elif 'right-button' in trigger:
                globals.x0 += segment_size
                globals.x1 += segment_size

            updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

            return updated_fig, globals.current_plot_index
        else:
            return current_fig, 0

    # Update plot when segment_size is changed
    @app.callback(
        Output('EEG-graph', 'figure', allow_duplicate=True),
        Input('segment-size', 'value'),
        [State('show-annotations-only', 'value'), State('use-slider', 'value'), State('annotation-label', 'value'), State('EEG-graph', 'figure')],
        prevent_initial_call=True
    )
    def _use_segment_slider(segment_size, show_annotations_only, use_slider, annotation_label, current_fig):
        """Moves viewed segment. Triggered when segment-slider is moved and when left- or right-arrow button is clicked.

        Args:
            segment_size (int): Segment size of EEG plot.
            show_annotations_only (bool): Whether or not to only show annotations.

        Returns:
            tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
        """
        if globals.plotting_data:
            if segment_size:
                globals.x1 = globals.x0 + segment_size + 1
            else:
                globals.x1 = (globals.raw.n_times / globals.raw.info['sfreq']) + 0.5

            updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)

            return updated_fig

        else:
            return current_fig