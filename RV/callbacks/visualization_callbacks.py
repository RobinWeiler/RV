import os
import math
from PIL import Image

from dash import Patch, ALL
from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
import plotly.express as px

import mne

import numpy as np
import pandas as pd

import RV.constants as c
from RV.callbacks.utils.annotation_utils import merge_annotations, annotations_to_raw
from RV.callbacks.utils.bad_channels_utils import bad_channel_disagrees, get_automatic_bad_channels
from RV.callbacks.utils.loading_utils import load_raw
from RV.callbacks.utils.preprocessing_utils import preprocess_EEG
from RV.callbacks.utils.visualization_utils import get_y_axis_ticks


def register_visualization_callbacks(disable_preprocessing=False):
    # Activate panning mode when RV-pan-button is clicked
    clientside_callback(
        """
            function(n_clicks) {
                document.querySelector("a[data-val='pan']").click()
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-pan-button', 'n_clicks'),
        Input('RV-pan-button', 'n_clicks'),
        prevent_initial_call=True
    )

    # Activate zooming mode when RV-zoom-button is clicked
    clientside_callback(
        """
            function(n_clicks) {
                document.querySelector("a[data-val='zoom']").click()
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-zoom-button', 'n_clicks'),
        Input('RV-zoom-button', 'n_clicks'),
        prevent_initial_call=True
    )

    @callback(
        [
            Output('RV-main-graph', 'figure'),
            Output('RV-main-graph-resampler', 'data'),
            Output('RV-annotation-overview-graph', 'figure'),
            Output('RV-hide-bad-channels-button', 'n_clicks')
        ],
        Input('RV-clear-main-graph-button', 'n_clicks'),
        [
            State('RV-plotting-data', 'data'),
            State('RV-file-paths', 'data'),
            State('RV-segment-size-input', 'value'),
            State('RV-segment-slider', 'marks'),
        ],
        # prevent_initial_call=True,
        memoize=True,
    )
    def init_graphs(clear_main_graph, plotting_data, file_paths, segment_size, segment_slider_marks):
        """Loads title image into RV-main-graph as placeholder and clears RV-main-graph-resampler.
        Also initializes RV-annotation-overview-graph.
        """
        if clear_main_graph:
            print('Loading title image...')

            current_path = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_path)
            title_image = Image.open(os.path.join(parent_dir, 'assets/RV_title_image.png'))

            fig = px.imshow(title_image)

            fig.update_traces(
                hoverinfo='skip',
                hovertemplate=None
            )

            fig.update_layout(
                margin=dict(
                    autoexpand=False,
                    b=0,
                    l=0,
                    pad=0,
                    r=0,
                    t=0,
                ),
            )

            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)


            print('Initializing annotation overview...')

            annotation_fig = go.Figure()

            if segment_slider_marks != None:
                x_ticks = [(int(segment) * segment_size) for segment in segment_slider_marks.keys()]
                x_ticks = np.array(x_ticks)

                annotation_fig.add_trace(
                    go.Scatter(
                        marker=dict(size=8, color=c.PLOT_COLOR),
                        mode='markers+text',
                        text=x_ticks,
                        textfont=dict(size=10),
                        textposition='bottom center',
                        x=x_ticks,
                        y=np.repeat(0.9, len(x_ticks)),
                    ),
                )
            else:
                annotation_fig.add_trace(
                    go.Scatter(
                        x=None,
                        y=None,
                    ),
                )

            annotation_fig.update_layout(
                autosize=False,
                dragmode=False,
                margin=dict(
                    autoexpand=False,
                    l=0,
                    r=0,
                    b=0,
                    t=0,
                    pad=0,
                ),
                paper_bgcolor=c.BACKGROUND_COLOR,
                plot_bgcolor=c.PLOT_COLOR
            )

            if plotting_data:
                x_axis_range = (-0.5, plotting_data['recording_length'] + 0.5)
            else:
                # when RV is launched
                x_axis_range = (0, 1)

            annotation_fig.update_xaxes(
                autorange=False,
                range=x_axis_range,
                showgrid=False,
                zeroline=False
            )

            annotation_fig.update_yaxes(
                range=(0, 1),
                showgrid=False,
                zeroline=False
            )

            return fig, None, annotation_fig, 0

        raise PreventUpdate

    @callback(
        [
            Output('RV-raw', 'data', allow_duplicate=True),
            Output('RV-plotting-data', 'data', allow_duplicate=True),
            Output('RV-model-data', 'data', allow_duplicate=True),
            Output('RV-main-graph', 'figure', allow_duplicate=True), Output('RV-main-graph-resampler', 'data', allow_duplicate=True),
            Output('RV-segment-slider', 'value', allow_duplicate=True),
            Output('RV-bad-channels-dropdown', 'value', allow_duplicate=True),
        ],
        Input({'type': 'process-button', 'modal': 'RV-settings'}, 'n_clicks'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-model-data', 'data'),
            State('RV-file-selection-dropdown', 'value'),
            State('RV-high-pass-input', 'value'),
            State('RV-low-pass-input', 'value'),
            State('RV-notch-freq-input', 'value'),
            State('RV-reference-dropdown', 'value'),
            State('RV-scale-input', 'value'),
            State('RV-offset-input', 'value'),
            State('RV-reorder-channels', 'value'),
            State('RV-segment-size-input', 'value'),
            State('RV-resample-points-input', 'value'),
            State('RV-skip-hoverinfo', 'value'),
            State('RV-channel-selection-dropdown', 'value'),
            State('RV-bad-channels-dropdown', 'value'),
            State('RV-bad-channel-detection-dropdown', 'value'),
            State('RV-bad-channel-interpolation', 'value'),
            State('RV-annotation-file-selection-dropdown', 'value'),
            State('RV-annotation-label', 'value'),
            State('RV-show-annotation-labels', 'value'),
            State('RV-annotations-only-mode', 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'value'),
            State({'type': 'color-dropdown', 'label': ALL}, 'id'),
            State('RV-run-model', 'value'),
            State('RV-annotate-model', 'value'),
            State('RV-model-threshold-input', 'value')
        ],
        prevent_initial_call=True,
    )
    def preprocess_and_plot(plot_button,
                            raw,
                            plotting_data,
                            model_data,
                            selected_file,
                            high_pass,
                            low_pass,
                            notch_freq,
                            reference,
                            scale,
                            channel_offset,
                            reorder_channels,
                            segment_size,
                            resample_points,
                            skip_hoverinfo,
                            selected_channels,
                            selected_bad_channels,
                            bad_channel_detection,
                            bad_channel_interpolation,
                            selected_annotation_files,
                            annotation_label,
                            show_annotation_labels,
                            annotations_only_mode,
                            annotation_colors,
                            annotation_colors_ids,
                            run_internal_model,
                            annotate_model,
                            model_threshold
    ):
        """Preprocesses loaded EEG data and plots it in RV-main-graph based on selections made in settings modal.
        Also updates RV-bad-channels-dropdown, RV-segment-slider, and various internally stored variables.
        """

        if not selected_file:
            raise PreventUpdate

        if plot_button:
            annotations_copy = None
            # if no new bandpass filter will be applied, keep current annotations
            if len(raw.annotations) > 0:
                if disable_preprocessing or ((raw != None) and ((high_pass == raw.info['highpass']) and (low_pass == raw.info['lowpass']))):
                    print('Keeping annotations...')
                    annotations_copy = raw.annotations.copy()

            # Reload raw to apply filters from scratch
            raw = load_raw(selected_file)
            raw = raw.pick('data')
            print(raw.info)
            print(raw.annotations)

            # re-apply annotations copied above (if applicable)
            if annotations_copy:
                raw.set_annotations(annotations_copy)
                # print(raw.annotations)

            if selected_annotation_files:
                loaded_annotations = []
                for file_path in selected_annotation_files:
                    if '.csv' in file_path:
                        loaded_annotations_df = pd.read_csv(file_path)
                        
                        annotation_onsets = loaded_annotations_df['onset'].tolist()
                        annotation_durations = loaded_annotations_df['duration'].tolist()
                        annotation_descriptions = loaded_annotations_df['description'].tolist()

                        for annotation_index in range(len(annotation_onsets)):
                            loaded_annotations.append((annotation_onsets[annotation_index], annotation_durations[annotation_index], annotation_descriptions[annotation_index]))
                    elif '.fif' in file_path:
                        loaded_annotations = mne.io.read_raw(file_path).annotations

                        for annotation_index in range(len(loaded_annotations.onset)):
                            loaded_annotations.append((loaded_annotations.onset[annotation_index], loaded_annotations.duration[annotation_index], loaded_annotations.description[annotation_index]))
                    else:
                        raise Exception('Unknown file type. Currently, only .csv and .fif files are supported for annotation files.')

                merged_annotations, _ = merge_annotations(loaded_annotations)
                raw = annotations_to_raw(merged_annotations, raw)

            if not disable_preprocessing:
                raw = preprocess_EEG(raw, high_pass, low_pass, notch_freq, reference)

            if bad_channel_detection == 'None':
                print('No automatic bad-channel detection')
                # bad_channel_detection = None
            else:
                print(f'Performing automatic bad channel detection using {bad_channel_detection}')
                detected_bad_channels = get_automatic_bad_channels(raw, bad_channel_detection)
                selected_bad_channels = list(set(detected_bad_channels + raw.info['bads']))

                # raw.info['bads'] = selected_bad_channels  # selected_bad_channels will be added to raw in bad_channels_callbacks.py
                plotting_data['bad_channels'][str(bad_channel_detection)] = detected_bad_channels

            if bad_channel_interpolation:
                print('Performing bad-channel interpolation')
                raw = raw.interpolate_bads(reset_bads=False, mode='accurate', method='spline', verbose=0)

            # if no channels are selected, all channels are selected
            if not selected_channels:
                selected_channels = raw.ch_names.copy()

            print(f'Channels to plot: {selected_channels}')
            # print(len(selected_channels))

            data_subset, times_subset = raw[selected_channels, :]
            if scale:
                data_subset *= (c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS * scale)
            else:
                data_subset *= c.CONVERSION_VALUE_VOLTS_TO_MICROVOLTS

            if not skip_hoverinfo:
                custom_data = data_subset.copy()

            if reorder_channels and all(channel_name in c.EGI129_CHANNELS for channel_name in selected_channels):
                channel_order = [selected_channels.index(channel_name) for channel_name in c.EGI129_CHANNELS if channel_name in selected_channels]
                channel_order = np.array(channel_order)

                data_subset = data_subset[channel_order]
                selected_channels = [selected_channels[channel_index] for channel_index in channel_order]

            plotting_data['selected_channels'] = selected_channels

            # Offset channel traces along the y-axis
            y_axis_ticks = get_y_axis_ticks(selected_channels, channel_offset, reorder_channels)
            data_subset += y_axis_ticks.reshape(-1, 1)

            fig: FigureResampler = FigureResampler(
                go.Figure(),
                default_downsampler=MinMaxLTTB(parallel=True),
                resampled_trace_prefix_suffix=("<b style='color:#2bb1d6'>[R]</b> ", ''),
                show_mean_aggregation_size=False
            )

            # Plot channel traces
            for channel_index, channel_name in enumerate(selected_channels):
                if channel_name in selected_bad_channels:
                    if bad_channel_disagrees(channel_name, plotting_data['bad_channels']):
                        channel_color = c.BAD_CHANNEL_DISAGREE_COLOR
                    else:
                        channel_color = c.BAD_CHANNEL_COLOR
                else:
                    channel_color = 'black'

                lobe_name = None
                if reorder_channels and channel_name != 'Cz':
                    for lobe, channels in c.CHANNELS_TO_LOBES_EGI129.items():
                        if channel_name in channels:
                            lobe_name = lobe
                            break

                fig.add_trace(
                    go.Scattergl(
                        customdata=custom_data[channel_index] if not skip_hoverinfo else None,
                        hoverinfo='all' if not skip_hoverinfo else 'none',
                        hovertemplate='<b>%{fullData.name}</b> | Time = %{x:.2f} seconds, Amplitude = %{customdata:.2f} μV' + '<extra></extra>' if not skip_hoverinfo else '',
                        line=dict(width=0.8),
                        marker=dict(size=0.1),
                        mode='lines+markers',
                        name=channel_name + (f' {lobe_name}' if lobe_name else '')
                    ),
                    hf_marker_color=channel_color,
                    hf_x=times_subset,
                    hf_y=data_subset[channel_index, :],
                    max_n_samples=resample_points if resample_points else np.inf
                )

            if run_internal_model:
                from RV.model.run_model import run_model

                print('Running model...')
                model_raw = mne.io.Raw(selected_file, preload=True, verbose=True)
                model_raw.info['bads'] = selected_bad_channels
                model_predictions, model_channel_names = run_model(model_raw)
                model_data['M0'] = {'predictions': model_predictions, 'channel_names': model_channel_names, 'annotation_description': 'annotation_M0'}

            model_counter = 0
            all_model_annotations = []
            if model_data != None:
                # Plot model predictions
                for model_name, model in model_data.items():
                    model_counter += 1
                    y_axis_ticks = np.append(y_axis_ticks, (-2 * c.DEFAULT_Y_AXIS_OFFSET * model_counter))

                    model_prediction_times = np.linspace(0, plotting_data['recording_length'], len(model['predictions']))

                    fig.add_trace(
                        go.Scattergl(
                            marker=dict(
                                size=10,
                                cmax=1,
                                cmin=0,
                                color=model['predictions'],
                                colorscale='RdBu_r'
                            ),
                            name=model_name,
                            mode='markers',
                            customdata=model['predictions'] if not skip_hoverinfo else None,
                            hoverinfo='all' if not skip_hoverinfo else 'none',
                            hovertemplate='Time=%{x:.2f}, Prediction=%{customdata:.2f}<extra><b>%{fullData.name}</b></extra>' if not skip_hoverinfo else ''
                        ),
                        hf_x=model_prediction_times,
                        hf_y=model['predictions'] + y_axis_ticks[-1],
                        max_n_samples=resample_points if resample_points else np.inf
                    )

                    if annotate_model and model_threshold:
                        predictions_greater_equal_threshold = np.greater_equal(model['predictions'], model_threshold)

                        model_annotations = []
                        model_annotation_onset = None
                        for prediction_index, prediction in enumerate(predictions_greater_equal_threshold):
                            if prediction:  # if prediction >= model_threshold
                                if model_annotation_onset is None:  # if new annotation starts, record onset (else next prediction)
                                    prediction_time = model_prediction_times[prediction_index]
                                    model_annotation_onset = prediction_time
                            elif model_annotation_onset != None:  # if prediction < model_threshold and previous prediction >= model_threshold, create annotation
                                prediction_time = model_prediction_times[prediction_index]
                                model_annotations.append((model_annotation_onset, prediction_time - model_annotation_onset, model['annotation_description']))
                                model_annotation_onset = None
                            else:
                                continue

                        # if last model annotation goes until end of recording
                        if model_annotation_onset is not None:
                            model_annotations.append((model_annotation_onset, model_prediction_times[-1] - model_annotation_onset, model['annotation_description']))

                        all_model_annotations += model_annotations

                if annotate_model and len(all_model_annotations) > 0:
                    loaded_annotations = []
                    for annotation_index in range(len(raw.annotations)):
                        loaded_annotations.append((raw.annotations.onset[annotation_index], raw.annotations.duration[annotation_index], raw.annotations.description[annotation_index]))

                    merged_annotations, _ = merge_annotations(loaded_annotations + all_model_annotations)
                    print(f'Current annotations: {merged_annotations}')

                    raw = annotations_to_raw(merged_annotations, raw)

            # Draw annotation boxes
            for annotation_index in range(len(raw.annotations)):
                for i, dropdown in enumerate(annotation_colors_ids):
                    if dropdown['label'] == raw.annotations.description[annotation_index]:
                        annotation_color = annotation_colors[i]
                        break

                fig.add_vrect(
                    editable=True,
                    fillcolor=annotation_color if annotation_color != 'hide' else 'red',
                    label={'text': raw.annotations.description[annotation_index], 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}} if show_annotation_labels else {},
                    layer='below',
                    line_width=0,
                    name=raw.annotations.description[annotation_index],
                    opacity=0.5,
                    visible=True if annotation_color != 'hide' else False,
                    x0=raw.annotations.onset[annotation_index],
                    x1=raw.annotations.onset[annotation_index] + raw.annotations.duration[annotation_index]
                )

            # Get longest channel name for margins
            longest_channel_name_length = len(max(selected_channels, key=len))
            longest_lobe_name_length = 0
            if reorder_channels:
                longest_lobe_name_length = len(max(c.CHANNELS_TO_LOBES_EGI129.keys()))

            # Get current annotation color for annotation drawing
            for index, dropdown in enumerate(annotation_colors_ids):
                if dropdown['label'] == annotation_label:
                    annotation_color = annotation_colors[index]
                    break

            fig.update_layout(
                autosize=False,
                dragmode='drawrect',
                legend=dict(x=1.01),
                margin=dict(
                    autoexpand=False,
                    b=20,  # for x-axis ticks
                    l=longest_channel_name_length * 6 + 15,  # for y-axis ticks (channel names)
                    pad=5,
                    r=(longest_channel_name_length + longest_lobe_name_length) * 6 + 110,  # for legend
                    t=5,  # for style
                ),
                newshape=dict(
                    fillcolor=annotation_color if annotation_color != 'hide' else 'red',
                    opacity=0.5,
                    drawdirection='vertical',
                    layer='below',
                    line_width=0,
                    label={'text': annotation_label, 'textposition': 'top center', 'font': {'size': 18, 'color': 'black'}} if show_annotation_labels else {},
                    name=annotation_label,
                    visible=True if annotation_color != 'hide' else False
                    # showlegend=True,
                    # legend='legend',
                    # legendgroup=annotation_label,
                    # legendgrouptitle={'text':annotation_label}
                ),
                paper_bgcolor=c.BACKGROUND_COLOR,
                plot_bgcolor=c.PLOT_COLOR,
            )

            # Set up x-axis
            if annotations_only_mode and len(raw.annotations) > 0:
                x_axis_range_0 = raw.annotations.onset[0] - 1
                x_axis_range_1 = raw.annotations.onset[0] + raw.annotations.duration[0] + 1
            else:
                x_axis_range_0 = -0.5
                if segment_size:
                    x_axis_range_1 = segment_size + 0.5
                else:
                    x_axis_range_1 = plotting_data['recording_length'] + 0.5

            x_axis_range = (x_axis_range_0, x_axis_range_1)

            fig.update_xaxes(
                maxallowed=0.5 + (math.ceil(plotting_data['recording_length'] / segment_size) * segment_size) if segment_size else plotting_data['recording_length'],
                minallowed=-0.5,
                range=x_axis_range,
                showgrid=True,
                zeroline=False
            )

            # Set up y-axis
            if channel_offset != 0:
                # spread traces from bottom to top
                y_axis_range_0 = np.min(y_axis_ticks) - (2 * c.DEFAULT_Y_AXIS_OFFSET)
            else:
                # position collapsed traces in the middle of the view
                y_axis_range_0 = -(len(selected_channels) + 2) * c.DEFAULT_Y_AXIS_OFFSET
            y_axis_range_1 = ((len(selected_channels) + 2) * c.DEFAULT_Y_AXIS_OFFSET)

            if reorder_channels:
                y_axis_range_1 += c.DEFAULT_Y_AXIS_OFFSET * ((len(c.CHANNELS_TO_LOBES_EGI129) - 1) * 2)

            if show_annotation_labels:
                y_axis_range_1 += 4 * (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)

            y_axis_range = (y_axis_range_0, y_axis_range_1)

            y_axis_labels = selected_channels.copy()
            if model_data != None:
                for model_name in model_data.keys():
                    y_axis_labels.append(model_name)

            fig.update_yaxes(
                range=y_axis_range,
                showgrid=False,
                tickfont=dict(size=12 if len(selected_channels) <= 64 else 8),
                tickmode='array',
                ticktext=y_axis_labels,
                tickvals=y_axis_ticks,
                zeroline=False,
            )

            return Serverside(raw), plotting_data, Serverside(model_data), fig, Serverside(fig), 0, selected_bad_channels

        raise PreventUpdate

    @callback(
        Output('RV-main-graph', 'figure', allow_duplicate=True),
        Input('RV-main-graph', 'relayoutData'),
        State('RV-main-graph-resampler', 'data'),
        prevent_initial_call=True,
        memoize=True,
    )
    def relayout_graph(relayout_data, resampler):
        """Handles relayout events such as zooming and panning.
        """
        if not any('axis.range' in key for key in relayout_data.keys()):
            # relayout events not related to axes-range changes are handled by different callbacks
            raise PreventUpdate

        print(f'New axes ranges: {relayout_data}')

        # Toggle spikelines
        # if 'yaxis.showspikes' in relayout_data.keys() and relayout_data['yaxis.showspikes'] == True:
        #     patched_fig = Patch()
        #     patched_fig['layout']['yaxis']['showspikes'] = False

        #     return patched_fig

        return resampler.construct_update_data_patch(relayout_data)

    @callback(
        Output('RV-main-graph', 'figure', allow_duplicate=True),
        Input('RV-reset-view-button', 'n_clicks'),
        [
            State('RV-raw', 'data'),
            State('RV-plotting-data', 'data'),
            State('RV-main-graph-resampler', 'data'),
            State('RV-segment-slider', 'value'),
            State('RV-segment-size-input', 'value'),
            State('RV-offset-input', 'value'),
            State('RV-reorder-channels', 'value'),
            State('RV-annotations-only-mode', 'value'),
            State('RV-show-annotation-labels', 'value'),
        ],
        prevent_initial_call=True
    )
    def reset_zoom(reset_view, raw, plotting_data, resampler, segment_slider, segment_size, channel_offset, reorder_channels, annotations_only_mode, show_annotation_labels):
        if resampler is None:
            raise PreventUpdate

        if reset_view:
            if annotations_only_mode:
                x_axis_range_0 = raw.annotations.onset[segment_slider] - 1
                x_axis_range_1 = raw.annotations.onset[segment_slider] + raw.annotations.duration[segment_slider] + 1
            else:
                if segment_size:
                    x_axis_range_0 = (segment_slider * segment_size) - 0.5
                    x_axis_range_1 = ((segment_slider * segment_size) + segment_size + 0.5)
                else:
                    x_axis_range_0 = -0.5
                    x_axis_range_1 = plotting_data['recording_length'] + 0.5

            if channel_offset != 0:
                # spread traces from bottom to top
                y_axis_range_0 = np.min(resampler['layout']['yaxis']['tickvals']) - (2 * c.DEFAULT_Y_AXIS_OFFSET)
            else:
                # position collapsed traces in the middle of the view
                y_axis_range_0 = -(len(plotting_data['selected_channels']) + 2) * c.DEFAULT_Y_AXIS_OFFSET
            y_axis_range_1 = ((len(plotting_data['selected_channels']) + 2) * c.DEFAULT_Y_AXIS_OFFSET)

            if reorder_channels:
                y_axis_range_1 += c.DEFAULT_Y_AXIS_OFFSET * ((len(c.CHANNELS_TO_LOBES_EGI129) - 1) * 2)

            if show_annotation_labels:
                y_axis_range_1 += 4 * (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)

            relayout_data = {'xaxis.range[0]': x_axis_range_0, 'xaxis.range[1]': x_axis_range_1, 'yaxis.range[0]': y_axis_range_0, 'yaxis.range[1]': y_axis_range_1}

            print(f'Resetting view to {relayout_data}')

            patched_fig = resampler.construct_update_data_patch(relayout_data)
            patched_fig['layout']['xaxis']['range'] = (x_axis_range_0, x_axis_range_1)
            patched_fig['layout']['yaxis']['range'] = (y_axis_range_0, y_axis_range_1)

            return patched_fig

        raise PreventUpdate

    @callback(
        Output('RV-annotation-overview-graph', 'figure', allow_duplicate=True),
        Input('RV-main-graph', 'figure'),
        [
            State('RV-main-graph-resampler', 'data'),
            State('RV-annotation-overview-graph', 'figure'),
            State('RV-segment-size-input', 'value')
        ],
        prevent_initial_call=True
    )
    def update_annotation_overview(current_fig, resampler, current_annotation_fig, segment_size):
        """Updates RV-annotation-overview-graph to include all annotations of main-graph as well as highlight where currently viewed segment is.
        """
        if resampler is None:
            raise PreventUpdate

        patched_annotation_fig = Patch()

        if 'shapes' in current_fig['layout'].keys():
            current_shapes = current_fig['layout']['shapes']
            for shape in current_shapes:
                # Change these attributes for annotations in RV-annotation-overview-graph
                shape['editable'] = False
                shape['label'] = {}
                shape['layer'] = 'above'
            patched_annotation_fig['layout']['shapes'] = current_shapes
        else:
            patched_annotation_fig['layout']['shapes'] = []

        if segment_size:
            # Highlight currently plotted segment with dark rectangles left and right
            patched_annotation_fig['layout']['shapes'].append({
                'editable': False,
                'fillcolor': 'black',
                'layer': 'above',
                'line': {'width': 0},
                'opacity': 0.3,
                'type': 'rect',
                'x0': current_annotation_fig['layout']['xaxis']['range'][0],
                'x1': current_fig['layout']['xaxis']['range'][0],
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y'
            })
            patched_annotation_fig['layout']['shapes'].append({
                'editable': False,
                'fillcolor': 'black',
                'layer': 'above',
                'line': {'width': 0},
                'opacity': 0.3,
                'type': 'rect',
                'x0': current_fig['layout']['xaxis']['range'][1],
                'x1': current_annotation_fig['layout']['xaxis']['range'][1],
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y'
            })

        return patched_annotation_fig
