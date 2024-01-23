from dash import Input, Output, State, Patch
from dash.exceptions import PreventUpdate

from plotly.graph_objs import Figure

from helperfunctions.channel_selection_helperfunctions import get_channel_locations_plot
from helperfunctions.modal_helperfunctions import _toggle_modal
from helperfunctions.visualization_helperfunctions import _get_y_ticks

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
        Input("modal-channel-select", "is_open"),
    )
    def _generate_channel_selection_plot(channel_selection_is_open):
        """Loads channel-topography plot if available. Triggers when new file is loaded.

        Args:
            file_name (string): Loaded EEG file-name.

        Returns:
            plotly.graph_objs.Figure: Figure with channel-topography plot.
        """
        if channel_selection_is_open:
            if globals.raw:
                topography_plot = get_channel_locations_plot(globals.raw)
            else:
                topography_plot = Figure()

            return topography_plot
        else:
            raise PreventUpdate
    
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
    #     [State('reorder-channels', 'value'), State('skip-hoverinfo', 'value')],
    #     prevent_initial_call=True
    # )
    # def _update_plotted_channels(selected_channels, reorder_channels, skip_hoverinfo):
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
    #     if globals.plotting_data['EEG']:
    #         check = all(channel in globals.raw.ch_names for channel in selected_channels)
    #         if check:
    #             updated_fig = Patch()

    #             globals.plotting_data['EEG']['channel_names'] = selected_channels
    #             globals.plotting_data['plot']['y_ticks'] = _get_y_ticks(globals.plotting_data, reorder_channels)

    #             index_0 = globals.viewing_raw.time_as_index(globals.x0)[0] if globals.x0 > 0 else 0
    #             index_1 = globals.viewing_raw.time_as_index(globals.x1)[0]

    #             data_subset, times_subset = globals.viewing_raw[globals.plotting_data['EEG']['channel_names'], index_0:index_1]
    #             data_subset = data_subset * globals.plotting_data['EEG']['scaling_factor']

    #             if not skip_hoverinfo:
    #                 custom_data = data_subset.copy()

    #             if len(globals.plotting_data['model']) > 0:
    #                 data_subset += globals.plotting_data['plot']['y_ticks'].reshape(-1, 1)[:-len(globals.plotting_data['model'])]
    #             else:
    #                 data_subset += globals.plotting_data['plot']['y_ticks'].reshape(-1, 1)

    #             updated_fig['data'] = []

    #             for channel_index in range(len(globals.plotting_data['EEG']['channel_names'])):
    #                 updated_fig['data'].append({})
    #                 updated_fig['data'][channel_index]['x'] = times_subset
    #                 updated_fig['data'][channel_index]['y'] = data_subset[channel_index, :]

    #                 if not skip_hoverinfo:
    #                     updated_fig['data'][channel_index]['customdata'] = custom_data[channel_index]

    #             updated_fig['layout']['yaxis']['tickvals'] = globals.plotting_data['plot']['y_ticks']

    #             return updated_fig
    #     else:
    #         raise PreventUpdate
