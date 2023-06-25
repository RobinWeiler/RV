from jupyter_dash import JupyterDash

from dash import dcc, html
import dash_bootstrap_components as dbc
# import dash_daq as daq

from plotly.graph_objs import Figure

from callbacks.channel_selection_callbacks import register_channel_selection_callbacks
from callbacks.modal_callbacks import register_modal_callbacks
from callbacks.loading_callbacks import register_loading_callbacks
from callbacks.saving_callbacks import register_saving_callbacks
from callbacks.annotation_callbacks import register_annotation_callbacks
from callbacks.bad_channel_callbacks import register_bad_channel_callbacks
from callbacks.model_callbacks import register_model_callbacks
from callbacks.segments_callbacks import register_segments_callbacks
from callbacks.stats_callbacks import register_stats_callbacks
from callbacks.visualization_callbacks import register_visualization_callbacks

def setup_app(disable_file_selection=False, disable_preprocessing=False):
    """Sets up HTML of RV application.

    Args:
        disable_file_selection (bool, optional): Whether or not to disable file selection. Defaults to False.
        disable_preprocessing (bool, optional): Whether or not to disable preprocessing options. Defaults to False.
    """
    app = JupyterDash(__name__, assets_folder='assets')
    app.layout = html.Div([
        # Top-bar buttons
        html.Div([
            html.Div([
                dbc.Button(
                    "Preprocessing",
                    id="open-file",
                    className='button',
                ),
                dbc.Button(
                    "Save to",
                    id="open-save",
                    className='button',
                    # style={'display':'none','weight':'0px','height':'0px'} if disable_file_selection else {}
                ),
            ], className='aligned first'),
            html.Div([
                html.Div([
                    dbc.Button(
                        "Rerun model",
                        id="rerun-model-button",
                        className='button'
                    ),
                ], className='aligned-threshold'),
                html.Div([
                    html.Font('Model threshold:', id='threshold-text-main')
                ], className='aligned-threshold'),
                html.Div([
                    dcc.Input(
                        id="model-threshold",
                        type='number',
                        value=0.7,
                        min=0,
                        max=1,
                        step=0.1,
                        debounce=True,
                        className='small-input'
                    ),
                ], className='aligned-threshold'),
            ], className='aligned second'),
            html.Div([
                dbc.Button(
                    "Annotation settings",
                    id="open-annotation-settings",
                    className='button',
                ),
                dbc.Button(
                    "Stats",
                    id="open-stats",
                    className='button'
                ),
                dbc.Button(
                    "Power spectrum",
                    id="open-power-spectrum",
                    className='button'
                ),
            ], className='aligned third', id='stats-buttons'),
            html.Div([
                dbc.Button(
                    "Help",
                    id="open-help",
                    className='button'
                ),
                dbc.Button(
                    "Quit",
                    id="quit-button",
                    className='button'
                ),
            ], className='aligned', id='right-buttons'),
        ], className='top-bar'),
        html.Div([
            html.Div([
                dbc.Button(
                    "<-",
                    id="left-button",
                    className='button'
                ),
            ], className='aligned'),
            
            html.Div([
                dbc.Button(
                    "-10s",
                    id="minus-ten-seconds-button",
                    className='button'
                ),
            ], className='aligned'),

            html.Div([
                dcc.Slider(0, 1, 1,
                    value=0,
                    disabled=True,
                    id='segment-slider',
                ),
            ], className='aligned', id='segment-slider-container'),
            
            html.Div([
                dbc.Button(
                    "+10s",
                    id="plus-ten-seconds-button",
                    className='button'
                ),
            ], className='aligned'),

            html.Div([
                dbc.Button(
                    "->",
                    id="right-button",
                    className='button'
                ),
            ], className='aligned')
        ], id='arrow-buttons'),

        # Open-file modal
        dbc.Modal([
            dbc.ModalHeader("Welcome to Robin's Viewer!"),
            dbc.ModalBody([
                # EEG-file upload
                html.Div([
                    html.Div([
                        html.H2('EEG-file selection'),
                        dcc.Upload(
                            id='upload-file',
                            children=html.Div([
                                'Drag-and-drop or ',
                                html.A('click here to select EEG file')
                            ]),
                            style={
                                'width': '97%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            } if not disable_file_selection else {'width': '0px', 'height': '0px', 'display': 'none'},
                            multiple=False
                        ),
                        html.Label([
                            'Selected file:',
                            html.Div(id='data-file')
                        ]),
                    ]),
                    html.Div([
                        dbc.Button(
                            "Stats",
                            id="open-stats-2",
                            className='button'
                        ),
                    ])
                ]),
                html.Hr(),

                # Pre-processing options
                html.Div([
                    html.H2('Preprocessing'),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Font('High-pass:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="high-pass",
                                    type='number',
                                    placeholder="in Hz",
                                    min=0,
                                    debounce=True,
                                    className='small-input',
                                    disabled=disable_preprocessing
                                )
                            ], className='aligned'),
                            html.Div([
                                html.Font('Low-pass:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="low-pass",
                                    type='number',
                                    placeholder="in Hz",
                                    debounce=True,
                                    className='small-input',
                                    disabled=disable_preprocessing
                                )
                            ], className='aligned')
                        ]),
                        html.Div([
                            html.Div([
                                html.Font('Reference:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Dropdown(
                                    id='reference-dropdown',
                                    options=[{'label': 'Average', 'value': 'average'}, {'label': 'Cz', 'value': 'Cz'}, {'label': 'None', 'value': 'None'}],
                                    # placeholder='Click here to select the reference...',
                                    value='None',
                                    clearable=False,
                                    className='small-dropdown',
                                    disabled=disable_preprocessing
                                )
                            ], className='aligned'),
                        ]),
                    ]),
                ]),
                html.Hr(),
                
                # Bad-channel options
                html.Div([
                    html.H2('Bad-channel handling'),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Font('Automatic bad-channel detection: ', className='header'),
                            ], className='aligned'),
                            html.Div([
                                dcc.Dropdown(
                                    id='bad-channel-detection-dropdown',
                                    options=[{'label': 'RANSAC', 'value': 'RANSAC'}, {'label': 'None', 'value': 'None'}],  # {'label': 'AutoReject', 'value': 'AutoReject'}
                                    value='None',
                                    clearable=False,
                                    className='small-dropdown'
                                )
                            ], className='aligned'),
                        ]),
                        html.Div([
                            html.Div([
                                html.Font('Bad channels: ', className='header'),
                            ], className='aligned'),
                            html.Div([
                                dcc.Dropdown(
                                    id='bad-channels-dropdown',
                                    multi=True,
                                    placeholder='Click here to select bad channels...',
                                )
                            ])
                        ]),
                        html.Div([
                            dbc.Checklist(
                                id='bad-channel-interpolation',
                                switch=True,  # no effect in Safari
                                options=[
                                    {'label': 'Interpolate bad channels', 'value': 1},
                                ],
                            )
                        ]),
                    ])
                ]),
                html.Hr(),

                # Visualization options
                html.Div([
                    html.H2('Visualization'),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Font('Resampling rate:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="resample-rate",
                                    type='number',
                                    placeholder="in Hz",
                                    min=0,
                                    debounce=True,
                                    className='small-input',
                                    disabled=disable_preprocessing
                                ),
                            ], className='aligned'),
                        ]),
                        html.Div([
                            html.Div([
                                html.Font('Scale:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="scale",
                                    type='number',
                                    placeholder="default: 1e-6",
                                    min=0,
                                    debounce=True,
                                    className='medium-input'
                                ),
                            ], className='aligned'),
                        ]),
                        html.Div([
                            html.Div([
                                html.Font('Channel-offset:', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="channel-offset",
                                    type='number',
                                    placeholder="default: 40 (μV)",
                                    min=0,
                                    debounce=True,
                                    className='medium-input'
                                ),
                            ], className='aligned'),
                        ]),
                    ], className='aligned'),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Font('Segments (in seconds):', className='header')
                            ], className='aligned'),
                            html.Div([
                                dcc.Input(
                                    id="segment-size",
                                    type='number',
                                    placeholder="No segments",
                                    min=1,
                                    debounce=True,
                                    className='medium-input',
                                ),
                            ], className='aligned'),
                        ]),
                        html.Div([
                            dbc.Checklist(
                                id='use-slider',
                                switch=True,  # no effect in Safari
                                options=[
                                    {'label': 'Activate view-slider', 'value': 1},
                                ],
                                value=[1]
                            )
                        ]),
                        html.Div([
                            dbc.Button(
                                "Select channels",
                                id="open-channel-select",
                                className='button'
                            ),
                        ]),
                    ], className='aligned right-pre'),
                ]),
                html.Hr(),
                
                # Deep-learning model & annotations
                html.Div([
                    html.Div([
                        html.H2('Annotations & deep-learning model'),
                        dcc.Upload(
                            id='upload-model-output',
                            children=html.Div([
                                'Drag-and-drop or ',
                                html.A('click here to select model-output files or annotation files')
                            ]),
                            style={
                                'width': '97%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=True
                        ),
                        html.Label([
                            'Selected files:',
                            html.Div(id='model-output-files')
                        ]),
                    ]),
                    html.Div([
                        dbc.Button("Remove model predictions", id="reset-models", className="button")
                    ]),
                    html.Div([
                        dbc.Checklist(
                            id='run-model',
                            switch=True,  # no effect in Safari
                            options=[
                                {'label': 'Run model from run_model.py', 'value': 0},
                            ],
                        )
                    ]),
                    html.Div([
                        html.Div([
                            dbc.Checklist(
                                id='annotate-model',
                                switch=True,  # no effect in Safari
                                options=[
                                    {'label': 'Annotate according to model(s)', 'value': 0}
                                ],
                            )
                        ], className='aligned checklist'),
                        # html.Div([
                        #     dcc.Input(
                        #         id="model-threshold",
                        #         type='number',
                        #         placeholder="default: 0.7",
                        #         min=0,
                        #         max=1,
                        #         step=0.1,
                        #         debounce=True,
                        #         className='medium-input'
                        #     ),
                        # ], className='aligned'),
                        html.Div([
                            dbc.Checklist(
                                id='show-annotations-only',
                                switch=True,  # no effect in Safari
                                options=[
                                    {'label': 'Only show annotations', 'value': 0},
                                ],
                            )
                        ]),
                    ]),
                ]),
                html.Hr(),

                dbc.Button("Plot", id="plot-button", className=['button'])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-file", className=["close-button", 'button'])
            )],
            id="modal-file",
            is_open=True,
            size='lg',
            centered=True,
        ),
        
        # Channel modal
        dbc.Modal([
            dbc.ModalHeader('Channel selection'),
            dbc.ModalBody([
                html.Div([
                    dcc.Dropdown(
                        id='selected-channels-dropdown',
                        multi=True,
                        placeholder='Click here to select channels to plot or select them in the plot below...',
                    )
                ]),
                html.Div([
                    dcc.Graph(
                        id='channel-topography',
                        figure=Figure(),
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': [
                                # 'lasso2d',
                                'autoScale2d',
                                'toImage',
                                'hoverClosestCartesian',
                                'hoverCompareCartesian',
                                'toggleSpikelines'
                            ],
                            'scrollZoom': True,
                        },
                        style={
                            'height': '70vh',
                        }
                    ),
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-channel-select", className=["close-button", 'button'])
            )],
            id="modal-channel-select",
            scrollable=False,
            is_open=False,
            size='lg',
            centered=True
        ),
        
        # Save modal
        dbc.Modal([
            dbc.ModalHeader('Save to'),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        html.Font('Save-file name:', className='header'),
                    ], className='aligned'),
                    html.Div([
                        dcc.Input(
                            id="save-file-name",
                            type='text',
                            placeholder="Enter file-name here",
                            debounce=True
                        ),
                    ], className='aligned'),
                    html.Div([
                        dcc.Dropdown(
                            id='extension-selection-dropdown',
                            options=[{'label': '.fif', 'value': '.fif'}],#, {'label': '.edf', 'value': '.edf'}, {'label': '.set (EEGLAB)', 'value': '.set'}],
                            # placeholder='Select file extension (default .fif)',
                            value='.fif',
                        ),
                    ], className='aligned'),
                ], id='save-to-input'),
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("Save with entered name", id="save-button", className=['button']),
                        dbc.Button("Overwrite current file", id="open-overwrite-button", className=['button']),
                        dbc.Button("Save annotations with entered name (.csv)", id="save-annotations-button", className=['button']),
                        dbc.Button("Save bad channels with entered name (.txt)", id="save-bad-channels-button", className=['button'])
                    ])
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-save", className=["close-button", 'button'])
            )],
            id="modal-save",
            is_open=False,
            size='lg',
            centered=True
        ),

        # Overwrite modal
        dbc.Modal([
            dbc.ModalHeader('Caution!'),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        html.Font('Are you sure you want to overwrite the current save-file?')
                    ]),
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("Yes", id="overwrite-button", className=['button']),
                            dbc.Button("No", id="cancel-overwrite-button", className=['button']),
                        ])
                    ]),
                ])
            ])],
            id="modal-overwrite",
            is_open=False,
            size='lg',
            centered=True
        ),
        
        # Annotation settings modal
        dbc.Modal([
            dbc.ModalHeader('Annotation settings'),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        dcc.RadioItems(
                            id='annotation-label',
                            options=[
                                {'label': 'bad_artifact', 'value': 'bad_artifact'},
                                {'label': 'bad_artifact_model', 'value': 'bad_artifact_model'},
                                # {'label': 'bad_drowsiness', 'value': 'bad_drowsiness'},
                            ],
                            value='bad_artifact',
                            labelStyle={'display': 'block'},
                        )
                    ], className='aligned'),
                    html.Div([
                        # daq.ColorPicker(
                        #     id="annotation-label-color",
                        #     label='Label color',
                        #     size=256,
                        #     value={'rgb': {'r': 255, 'g': 0, 'b': 0, 'a': 1}}
                        # ),
                        dcc.Dropdown(
                            id='annotation-label-color',
                            options=[
                                {'label': 'red', 'value': 'red'},
                                {'label': 'green', 'value': 'green'},
                                {'label': 'blue', 'value': 'blue'},
                                {'label': 'yellow', 'value': 'orange'},
                                {'label': 'turquoise', 'value': 'turquoise'},
                                {'label': 'purple', 'value': 'purple'}
                            ],
                            value='red',
                            clearable=False,
                            className='small-dropdown'
                        )
                    ], className='aligned'),
                ]),
                html.Div([
                    dcc.Input(
                        id="new-annotation-label",
                        placeholder="New annotation label",
                        debounce=True,
                        minLength=1
                    ),
                ]),
                html.Div([
                    dbc.Button("Remove last label", id="remove-annotation-label", className=['button'])
                ]),
                # html.Div([
                #     dbc.Checklist(
                #         id='show-annotation-labels',
                #         switch=True,  # no effect in Safari
                #         options=[
                #             {'label': 'Display annotation labels', 'value': 1},
                #         ],
                #     )
                # ]),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-annotation-settings", className=["close-button", 'button'])
            )],
            id="modal-annotation-settings",
            is_open=False,
            size='lg',
            centered=True
        ),

        # Stats modal
        dbc.Modal([
            dbc.ModalHeader('Statistics'),
            dbc.ModalBody([
                html.Div([
                    html.H2('File name:'),
                    html.Font(id='file-name')
                ]),
                html.Div([
                    html.H2('Recording length (in seconds):'),
                    html.Font(id='recording-length')
                ]),
                html.Div([
                    html.H2('Amount of annotated data (in seconds):'),
                    html.Font(id='#noisy-data')
                ]),
                html.Div([
                    html.H2('Amount of clean data left (in seconds):'),
                    html.Font(id='#clean-data')
                ]),
                html.Div([
                    html.H2('Amount of clean intervals longer than 2 seconds:'),
                    html.Font(id='#clean-intervals')
                ]),
                html.Div([
                    dcc.Graph(
                        id='clean-intervals-graph',
                        figure=Figure(),
                        config={
                            'displayModeBar': False,
                        },
                    ),
                ]),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-stats", className=["close-button", 'button'])
            )],
            id="modal-stats",
            scrollable=True,
            is_open=False,
            size='lg',
            centered=True
        ),
        
        # Power-spectrum modal
        dbc.Modal([
            dbc.ModalHeader('Power spectrum of selected interval'),
            dbc.ModalBody([
                html.Div([
                    html.H2('Most prominent frequency:'),
                    html.Font(id='selected-data'),
                ]),
                html.Div([
                    html.H2('Welch power spectrum:'),
                    dcc.Graph(
                        id='power-spectrum',
                        figure=Figure(),
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True
                        },
                    ),
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-power-spectrum", className=["close-button", 'button'])
            )],
            id="modal-power-spectrum",
            scrollable=True,
            is_open=False,
            size='lg',
            centered=True
        ),

        # Help modal
        dbc.Modal([
            dbc.ModalHeader('Help'),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        html.Label([
                            'Publication: ',
                            html.A(
                                'Click here',
                                href='https://doi.org/10.3389/fninf.2022.1025847',
                                target='_blank'
                            )
                        ]),
                    ]),
                    html.Div([
                        html.Label([
                            'GitHub: ',
                            html.A(
                                'Click here',
                                href='https://github.com/RobinWeiler/RV',
                                target='_blank'
                            )
                        ]),
                    ]),
                    html.Div([
                        html.Label([
                            'Documentation: ',
                            html.A(
                                'Click here',
                                href='https://robinweiler.github.io/RV/RV.html',
                                target='_blank'
                            )
                        ])
                    ]),
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-help", className=["close-button", 'button'])
            ),
            ],
            id="modal-help",
            is_open=False,
            size='lg',
            centered=True
        ),

        # Quit modal
        dbc.Modal([
            dbc.ModalHeader('Caution!'),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        html.Font('Are you sure you want to quit? Data will be saved to save_file_path if applicable.')
                    ]),
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("Yes", id="final-quit-button", className=['button']),
                            dbc.Button("No", id="cancel-quit-button", className=['button']),
                        ])
                    ]),
                ])
            ])],
            id="modal-quit",
            is_open=False,
            size='lg',
            centered=True
        ),

        # EEG graph
        dcc.Loading(
            id="loading-icon",
            children=[
                dcc.Graph(
                    id='EEG-graph',
                    figure=Figure(),
                    config={
                        'modeBarButtonsToAdd':[
                            'drawrect',
                            'eraseshape',
                            'toggleSpikelines'
                        ],
                        'modeBarButtonsToRemove':[
                            'lasso2d',
                            'autoScale2d',
                            'hoverClosestCartesian',
                            'hoverCompareCartesian',
                        ],
                        'displayModeBar': True,
                        # 'doubleClick': 'reset'
                        'doubleClickDelay': 300,
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'png', # one of png, svg, jpeg, webp
                            'filename': 'EEG_RV',
                        },
                        'scrollZoom': True
                    },
                    style={
                        'height': '90vh',
                    },
                ),
            ],
            type='default',
            # color='red',
            parent_className='loading_wrapper',
        ),

        # Hidden output variables
        html.Pre(id='relayout-data'),
        html.Pre(id='preload-data'),
        html.Pre(id='chosen-channels'),
        html.Pre(id='chosen-model'),
        html.Pre(id='chosen-model-threshold'),
        html.Pre(id='chosen-save-file-name'),
        html.Pre(id='chosen-extension'),
        html.Pre(id='chosen-overwrite'),
        html.Pre(id='chosen-annotation-color'),
        html.Pre(id='save-file'),
        html.Pre(id='save-annotations'),
        html.Pre(id='save-bad-channels'),
        html.Pre(id='overwrite-file'),
        html.Pre(id='quit-viewer'),
        html.Pre(id='quit-viewer-close')
    ])

    # Register callbacks
    register_channel_selection_callbacks(app)
    register_modal_callbacks(app)
    register_loading_callbacks(app)
    register_saving_callbacks(app)
    register_annotation_callbacks(app)
    register_bad_channel_callbacks(app)
    register_model_callbacks(app)
    register_segments_callbacks(app)
    register_stats_callbacks(app)
    register_visualization_callbacks(app)
    
    return app