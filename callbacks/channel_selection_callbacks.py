from dash.dependencies import Input, Output, State
from plotly.graph_objs import Figure

from helperfunctions.channel_selection_helperfunctions import get_channel_locations_plot
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.visualization_helperfunctions import get_EEG_plot

import globals

def register_channel_selection_callbacks(app):

    # Toggle channel-selection modal
    @app.callback(
        Output("modal-channel-select", "is_open"),
        [Input("open-channel-select", "n_clicks"), Input("close-channel-select", "n_clicks")],
        State("modal-channel-select", "is_open"),
        prevent_initial_call=True
    )
    def _toggle_channel_selection_modal(open_channel_select, close_channel_select, is_open):
        """Opens or closes channel-select modal based on relevant button clicks and loads channel-topography plot if available.

        Args:
            open_channel_select (int): Num clicks on open-channel-select button.
            close_channel_select (int): Num clicks on close-channel-select button.
            is_open (bool): Whether or not modal is currently open.

        Returns:
            bool: Bool whether or not modal should now be open.
        """
        return _toggle_modal([open_channel_select, close_channel_select], is_open)

    # Generate channel-selection plot
    @app.callback(
        Output("channel-topography", "figure"),
        Input('data-file', 'children'),
    )
    def _generate_channel_selection_plot(file_name):
        """Loads channel-topography plot if available. Triggers when new file is loaded.

        Args:
            file_name (string): Loaded EEG file-name.

        Returns:
            plotly.graph_objs.Figure: Figure with channel-topography plot.
        """

        if globals.raw:
            topography_plot = get_channel_locations_plot(globals.raw)
        else:
            topography_plot = Figure()

        return topography_plot
    
    # Selecting channels to plot
    @app.callback(
        Output('selected-channels-dropdown', 'value'),
        Input('channel-topography', 'selectedData'),
        prevent_initial_call=True
    )
    def _get_selected_channels(selectedData):
        """Retrieves names of selected channels. Triggered when datapoints are selected in channel-topography plot.

        Args:
            selectedData (dict): Data from latest selection event.

        Returns:
            list: List of strings of channels selected for plotting.
        """
        # print(json.dumps(selectedData, indent=2))
        selected_channels = []

        if selectedData:
            for selected_channel in selectedData['points']:
                # print(selected_channel['customdata'])
                selected_channels.append(selected_channel['customdata'])

        print(selected_channels)

        return selected_channels

    # # Update plot when channels to plot are selected
    # @app.callback(
    #     Output('EEG-graph', 'figure', allow_duplicate=True),
    #     Input('selected-channels-dropdown', 'value'),
    #     [State('show-annotations-only', 'value'), State('use-slider', 'value'), State('annotation-label', 'value'), State('EEG-graph', 'figure')],
    #     prevent_initial_call=True
    # )
    # def _update_plotted_channels(selected_channels, show_annotations_only, use_slider, annotation_label, current_fig):
    #     """Moves viewed segment. Triggered when segment-slider is moved and when left- or right-arrow button is clicked.

    #     Args:
    #         selected_channels (list): List of strings of channels selected for plotting.
    #         show_annotations_only (bool): Whether or not to only show annotations.
    #         use_slider (bool): Whether or not to activate view-slider.
    #         annotation_label (string); Label for new annotations.
    #         current_fig (plotly.graph_objs.Figure): The current EEG plot.

    #     Returns:
    #         tuple(plotly.graph_objs.Figure, int): New EEG-plot segment and segment-slider value.
    #     """
    #     if globals.plotting_data:

    #         updated_fig = get_EEG_plot(globals.plotting_data, globals.x0, globals.x1, annotation_label, use_slider, show_annotations_only)
    #         return updated_fig

    #     else:
    #         return current_fig