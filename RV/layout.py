import os

from dash import dcc, html, Input, Output, State, callback, MATCH
import dash_bootstrap_components as dbc

import RV.constants as c
from RV.callbacks.utils.annotation_utils import get_annotation_label_radioitem
from RV.callbacks.utils.loading_utils import get_file_selection_options


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_RV_layout(load_file_path=os.path.join(CURRENT_PATH, 'data'), save_file_path=os.path.join(CURRENT_PATH, 'save_files'), serverside_cache=os.path.join(CURRENT_PATH, 'file_system_backend'),
                  temp_save_file_path=os.path.join(CURRENT_PATH, 'temp_raw.fif'),
                  disable_file_selection=False, disable_manual_saving=False, disable_model=True, disable_preprocessing=False,
                  auto_save=False
):
    file_paths = {'load_file_path': load_file_path, 'save_file_path': save_file_path, 'temp_save_file_path': temp_save_file_path, 'serverside_cache': serverside_cache}

    layout = html.Div(id='RV-container',
        children=[
            ##### Internal storage #####
            dcc.Store(id='RV-file-paths', data=file_paths),
            dcc.Store(id='RV-raw'),
            dcc.Store(id='RV-plotting-data', data={}),
            dcc.Store(id='RV-model-data', data={}),
            dcc.Store(id='RV-main-graph-resampler'),

            ##### Hidden buttons #####
            html.Div(id='RV-hidden-buttons',
                style={'display': 'none'},
                children=[
                    html.Button(id='RV-refresh-annotations-button'),
                    html.Button(id='RV-clear-main-graph-button', n_clicks=1),
                    html.Button(id='RV-increase-scale-button'),
                    html.Button(id='RV-decrease-scale-button'),
                    html.Button(id='RV-increase-offset-button'),
                    html.Button(id='RV-decrease-offset-button'),
                    dcc.Slider(
                            id='RV-segment-slider',
                            min=0,
                            max=0,
                            step=1,
                            value=None,
                            # disabled=True,
                        ),
                ]
            ),

            ##### Main GUI #####
            html.Div(id='RV-main-GUI',
                children=[
                    ##### Menu bar #####
                    html.Div(id='RV-menu-bar', className='aligned',
                        children=[
                            html.Div(children=[
                                    html.Button(
                                        'Settings',
                                        id={'type': 'open-button', 'modal': 'RV-settings'},  # 'RV-open-settings-button',
                                        className='blue-button'
                                    ),
                                    html.Button(
                                        'Save to',
                                        id={'type': 'open-button', 'modal': 'RV-save'},  # 'RV-open-save-button',
                                        className='blue-button',
                                        style={'display': 'none'} if disable_manual_saving else {},
                                    ),
                                ]
                            ),
                            html.Div(id='RV-annotations-buttons',
                                children=[
                                    html.Button(
                                        'Annotate segment',
                                        id='RV-mark-annotations-button',
                                        className='active-button',
                                        n_clicks=0,
                                        title="Shortcut: 'a'"
                                    ),
                                    html.Button(
                                        'Delete annotation',
                                        id='RV-delete-annotation-button',
                                        n_clicks=0,
                                        title="Shortcut: 'backspace'"
                                    ),
                                ],
                            ),
                            html.Div(id='RV-bad-channel-buttons',
                                children=[
                                    html.Button(
                                        'Mark bad channels',
                                        id='RV-mark-bad-channels-button',
                                        className='inactive-button',
                                        n_clicks=0,
                                        title="Shortcut: 'b'"
                                    ),
                                    html.Button(
                                        'Hide bad channels',
                                        id='RV-hide-bad-channels-button',
                                        n_clicks=0,
                                        disabled=True,
                                        title="Click again to show bad channels"
                                    ),
                                ],
                            ),
                            html.Div(id='RV-power-spectrum-buttons',
                                children=[
                                    html.Button(
                                        'Select segment',
                                        id='RV-select-segment-button',
                                        className='inactive-button',
                                        n_clicks=0,
                                        title="Shortcut: 's'"
                                    ),
                                    html.Button(
                                        'Power spectrum',
                                        id={'type': 'open-button', 'modal': 'RV-psd'},  # 'RV-open-power-spectrum-button',
                                    ),
                                ]
                            ),
                            html.Div(id='RV-zoom-buttons',
                                children=[
                                    html.Button(
                                        'Pan',  # (drag & drop)
                                        id='RV-pan-button',
                                        className='inactive-button',
                                        n_clicks=0,
                                        title="Shortcut: 'p'"
                                    ),
                                    html.Button(
                                        'Zoom',  # (drag & drop)
                                        id='RV-zoom-button',
                                        className='inactive-button',
                                        n_clicks=0,
                                        title="Shortcut: 'z'"
                                    ),
                                    html.Button(
                                        'Reset view',
                                        id='RV-reset-view-button',
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            html.Div(id='RV-model-buttons',
                                style={'display': 'none'},
                                children=[
                                    html.Button(
                                        'Highlight model channels',
                                        id='RV-highlight-model-channels-button',
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.Button(
                                        'Stats',
                                        id={'type': 'open-button', 'modal': 'RV-stats'},  # 'RV-open-stats-button',
                                        disabled=True,
                                        n_clicks_timestamp=-1
                                    ),
                                    html.Button(
                                        'Help',
                                        id={'type': 'open-button', 'modal': 'RV-help'},  # 'RV-open-help-button',
                                        className='blue-button'
                                    ),
                                    html.Button(
                                        'Quit',
                                        id={'type': 'open-button', 'modal': 'RV-quit'},  # 'RV-open-quit-button',
                                        className='red-button'
                                        # style={'background-color': '#eb5f6e'}
                                    ),
                                ]
                            ),
                        ]
                    ),

                    ##### Main graph #####
                    html.Div(id='RV-main-graph-container',
                        children=[
                            dcc.Loading(
                                id='RV-main-graph-loading',
                                type='default',
                                # color='red',
                                style={
                                    'opacity': 0,
                                    'animation-name': 'fadeIn',
                                    'animation-duration': '100ms',
                                    'animation-timing-function': 'ease-out',
                                    'animation-delay': '1000ms',
                                    'animation-fill-mode': 'forwards',
                                },
                                parent_style={
                                    'height': '100%',
                                },
                                children=[
                                    dcc.Graph(
                                        id='RV-main-graph',
                                        config={
                                            'modeBarButtonsToAdd':[
                                                'drawrect',
                                                'eraseshape',
                                                'toggleSpikelines'
                                            ],
                                            'modeBarButtonsToRemove':[
                                                'zoomIn2d',
                                                'zoomOut2d',
                                                'autoScale2d',
                                                'lasso2d',
                                                'hoverClosestCartesian',
                                                'hoverCompareCartesian',
                                            ],
                                            'displaylogo': False,
                                            'displayModeBar': True,
                                            # 'doubleClick': 'reset'
                                            'doubleClickDelay': 300,
                                            # 'toImageButtonOptions': {
                                            #     'format': 'png', # one of png, svg, jpeg, webp
                                            #     'filename': 'EEG_RV',
                                            # },
                                            'scrollZoom': True
                                        },
                                    ),
                                ]
                            )
                        ]
                    ),

                    ##### Segment bar #####
                    html.Div(id='RV-segment-bar', className='aligned',
                        children=[
                            html.Div([
                                html.Button(
                                    '←',
                                    id='RV-left-arrow-button',
                                    className='blue-button arrow-button',
                                    disabled=True,
                                ),
                            ]),
                            html.Div(id='RV-annotation-overview-graph-container',
                                children=[
                                    dcc.Graph(
                                        id='RV-annotation-overview-graph',
                                        config={
                                            'displayModeBar': False,
                                            'staticPlot': False,
                                            'scrollZoom': False
                                        },
                                    ),
                                ]
                            ),
                            html.Div([
                                html.Button(
                                    '→',
                                    id='RV-right-arrow-button',
                                    className='blue-button arrow-button',
                                    disabled=True,
                                ),
                            ])
                        ]
                    ),
                ]
            ),

            ##### Settings modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-settings'},
                is_open=True,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(close_button=False,
                        children=[
                            html.Div([
                                dbc.ModalTitle("Welcome to Robin's Viewer!"),
                            ]),
                            html.Div([
                                html.Img(
                                    src='assets/RV_logo.png',
                                    alt='RV logo',
                                    style={'width': '30px'}
                                )
                            ])
                        ]
                    ),
                    dbc.ModalBody([
                        html.Div(id='RV-file-selection-container',
                            style={'display': 'none'} if disable_file_selection else {},
                            children=[
                                html.H1('File selection'),
                                html.Div([
                                    dcc.Dropdown(
                                        id='RV-file-selection-dropdown',
                                        options=get_file_selection_options(file_paths, ['.fif', '.raw', '.edf', '.bdf', '.set']),
                                        value=None if not disable_file_selection else temp_save_file_path,
                                        clearable=False,
                                        placeholder='Select file to load...'
                                    ),
                                ]),
                                html.Div([
                                    html.Button(
                                        'Stats',
                                        id='RV-open-stats-button-2',
                                        disabled=True,
                                        n_clicks_timestamp=-1
                                    ),
                                ]),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-preprocessing-settings-container',
                            style={'display': 'none'} if disable_preprocessing else {},
                            children=[
                                html.H1('Preprocessing settings'),
                                html.Div([
                                    html.Div(className='aligned',
                                        children=[
                                            html.Div([
                                                html.Span('High pass (in Hz):'),
                                                dcc.Input(
                                                    id='RV-high-pass-input',
                                                    className='medium-input',
                                                    type='number',
                                                    placeholder='No filter',
                                                    min=0,
                                                    debounce=True,
                                                )
                                            ]),
                                            html.Div([
                                                html.Span('Low pass (in Hz):'),
                                                dcc.Input(
                                                    id='RV-low-pass-input',
                                                    className='medium-input',
                                                    type='number',
                                                    min=0,
                                                    placeholder='No filter',
                                                    debounce=True,
                                                )
                                            ]),
                                        ]
                                    ),
                                    html.Div([
                                        html.Span('Notch frequency (in Hz):'),
                                        dcc.Input(
                                            id='RV-notch-freq-input',
                                            className='medium-input',
                                            type='number',
                                            min=0,
                                            placeholder='No filter',
                                            debounce=True,
                                        )
                                    ]),
                                    html.Div(className='aligned',
                                        children=[
                                            html.Span('Re-reference:'),
                                            dcc.Dropdown(
                                                id='RV-reference-dropdown',
                                                className='small-dropdown',
                                                options=[{'label': 'Average', 'value': 'average'}, {'label': 'Cz', 'value': 'Cz'}, {'label': 'None', 'value': 'None'}],
                                                # placeholder='Click here to select the reference...',
                                                value='None',
                                                clearable=False,
                                            )
                                        ]
                                    ),
                                ]),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-visualization-settings-container',
                            children=[
                                html.H1('Visualization settings'),
                                html.Div(className='aligned',
                                    children=[
                                        html.Div(id='RV-left-visualization-settings-container',
                                            children=[
                                                # html.H2('Visual'),
                                                html.Div([
                                                    html.Span('Scale:'),
                                                    dcc.Input(
                                                        id='RV-scale-input',
                                                        className='medium-input',
                                                        type='number',
                                                        placeholder='default: 1 μV',
                                                        min=0.1,
                                                        step=0.1,
                                                        debounce=True,
                                                    ),
                                                ]),
                                                html.Div([
                                                    html.Span('Channel-offset (in μV):'),
                                                    dcc.Input(
                                                        id='RV-offset-input',
                                                        className='medium-input',
                                                        type='number',
                                                        placeholder='default: 40 μV',
                                                        min=0,
                                                        step=1,
                                                        debounce=True,
                                                    ),
                                                ]),
                                                html.Div([
                                                    dcc.Checklist(
                                                        id='RV-reorder-channels',
                                                        inline=True,
                                                        options=[
                                                            {'label': 'Order channels according to lobes (only EGI-129)', 'value': 1},
                                                        ],
                                                        # value=[1]
                                                    )
                                                ]),
                                            ]
                                        ),
                                        html.Div(id='RV-right-visualization-settings-container',
                                            children=[
                                                # html.H2('Performance'),
                                                html.Div([
                                                    html.Span('Segments (in seconds):'),
                                                    dcc.Input(
                                                        id='RV-segment-size-input',
                                                        className='medium-input',
                                                        type='number',
                                                        placeholder='No segments',
                                                        value=c.DEFAULT_SEGMENT_SIZE,
                                                        debounce=True,
                                                    ),
                                                ]),
                                                html.Div([
                                                    html.Span('Points plotted (per trace):'),
                                                    dcc.Input(
                                                        id='RV-resample-points-input',
                                                        className='medium-input',
                                                        type='number',
                                                        placeholder='All points',
                                                        # value=1000,
                                                        debounce=True,
                                                    ),
                                                ]),
                                                html.Div([
                                                    dcc.Checklist(
                                                        id='RV-skip-hoverinfo',
                                                        inline=True,
                                                        options=[
                                                            {'label': 'Disable info on hover', 'value': 1},
                                                        ],
                                                        # value=[1]
                                                    )
                                                ]),
                                            ]
                                        ),
                                    ]
                                ),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-channel-selection-container',
                            children=[
                                html.H1('Channel selection'),
                                html.Div([
                                    html.Div([
                                        dcc.Dropdown(
                                            id='RV-channel-selection-dropdown',
                                            multi=True,
                                            placeholder='Click here to select channels to plot or select them in the plot below (by default all)...',
                                        )
                                    ]),
                                    html.Div([
                                        html.Button(
                                            'Select 10-20 channels',
                                            id='RV-10-20-channels-button',
                                        ),
                                    ]),
                                    html.Div(id='RV-channel-selection-graph-container',
                                        children=[
                                            dcc.Graph(
                                                id='RV-channel-selection-graph',
                                                style={'display': 'none'},
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
                                            ),
                                        ]
                                    )
                                ]),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-bad-channel-settings-container',
                            children=[
                                html.H1('Bad-channel settings'),
                                html.Div([
                                    html.Div([
                                        dcc.Dropdown(
                                            id='RV-bad-channel-file-selection-dropdown',
                                            options=get_file_selection_options(file_paths, ['.txt']),
                                            clearable=True,
                                            multi=True,
                                            placeholder='Select bad-channel files (.txt) to load...'
                                        )
                                    ]),
                                    html.Div(className='aligned',
                                        children=[
                                            html.Span('Automatic bad-channel detection:'),
                                            dcc.Dropdown(
                                                id='RV-bad-channel-detection-dropdown',
                                                className='small-dropdown',
                                                options=[{'label': 'RANSAC', 'value': 'RANSAC'}, {'label': 'None', 'value': 'None'}],  # {'label': 'AutoReject', 'value': 'AutoReject'}
                                                value='None',
                                                clearable=False,
                                            )
                                        ]
                                    ),
                                    html.Div([
                                        html.H2('Bad channels:'),
                                        dcc.Dropdown(
                                            id='RV-bad-channels-dropdown',
                                            multi=True,
                                            placeholder='Click here to select bad channels...',
                                        )
                                    ]),
                                    html.Div([
                                        dcc.Checklist(
                                            id='RV-bad-channel-interpolation',
                                            inline=True,
                                            options=[
                                                {'label': 'Interpolate bad channels', 'value': 1},
                                            ],
                                        )
                                    ]),
                                ]),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-annotation-settings-container',
                            children=[
                                html.H1('Annotation settings'),
                                html.Div([
                                    html.Div([
                                        dcc.Dropdown(
                                            id='RV-annotation-file-selection-dropdown',
                                            className='dropdown',
                                            options=get_file_selection_options(file_paths, ['.csv', '.fif']),
                                            clearable=True,
                                            multi=True,
                                            placeholder='Select annotation files (.csv, .fif) to load...'
                                        )
                                    ]),
                                    html.Div([
                                        html.Div([
                                            dcc.RadioItems(
                                                id='RV-annotation-label',
                                                options=
                                                    [
                                                        get_annotation_label_radioitem('bad_artifact', 'red')[0],
                                                    ],
                                                value='bad_artifact',
                                                labelStyle={'display': 'flex', 'flex-direction': 'row', 'vertical-align': 'middle'},
                                            )
                                        ]),
                                    ]),
                                    html.Div([
                                        dcc.Input(
                                            id='RV-new-annotation-label-input',
                                            placeholder='New annotation label',
                                            debounce=True,
                                            minLength=1,
                                            type='text'
                                        ),
                                    ]),
                                    html.Div(className='aligned',
                                        children=[
                                            html.Button(
                                                'Remove selected label',
                                                id={'type': 'open-button', 'modal': 'RV-remove-annotation-label'},
                                                disabled=True
                                            ),
                                            html.Button(
                                                'Rename selected label',
                                                id={'type': 'open-button', 'modal': 'RV-rename-annotation-label'},
                                            )
                                        ]
                                    ),
                                    html.Div([
                                        dcc.Checklist(
                                            id='RV-show-annotation-labels',
                                            inline=True,
                                            options=[
                                                {'label': 'Display annotation labels', 'value': 1},
                                            ],
                                        )
                                    ]),
                                    html.Div([
                                        dcc.Checklist(
                                            id='RV-annotations-only-mode',
                                            inline=True,
                                            options=[
                                                {'label': 'Only show annotations', 'value': 1},
                                            ],
                                        )
                                    ]),
                                ]),
                                html.Hr(),
                            ]
                        ),
                        html.Div(id='RV-model-settings-container',
                            style={'display': 'none'} if disable_model else {},
                            children=[
                                html.H1('Model settings'),
                                html.Div([
                                    html.Div([
                                        dcc.Dropdown(
                                            id='RV-model-file-selection-dropdown',
                                            className='dropdown',
                                            options=get_file_selection_options(file_paths, ['.txt', '.pt', '.npy']),
                                            clearable=True,
                                            multi=True,
                                            placeholder='Select model-prediction files (.txt, .pt, .npy) to load...'
                                        )
                                    ]),
                                    html.Div([
                                        dcc.Checklist(
                                            id='RV-run-model',
                                            inline=True,
                                            options=[
                                                {'label': 'Run model from run_model.py', 'value': 0},
                                            ],
                                        )
                                    ]),
                                    html.Div([
                                        html.Div([
                                            dcc.Checklist(
                                                id='RV-annotate-model',
                                                inline=True,
                                                options=[
                                                    {'label': 'Annotate according to model(s)', 'value': 0}
                                                ],
                                            )
                                        ]),
                                        html.Div([
                                            html.Span('Model threshold:'),
                                            dcc.Input(
                                                id='RV-model-threshold-input',
                                                className='large-input',
                                                type='number',
                                                placeholder='Annotate predictions above',
                                                min=0,
                                                max=1,
                                                value=0.7,
                                                step=0.1,
                                                debounce=True,
                                                disabled=True,
                                            ),
                                        ]),
                                    ]),
                                ]),
                                html.Hr(),
                            ]
                        ),
                    ]),
                    dbc.ModalFooter(class_name='binary-modal-footer',
                        children=[
                            html.Button('Plot', id={'type': 'process-button', 'modal': 'RV-settings'}, className='blue-button'),  # 'RV-plot-button'
                            html.Button('Close', id={'type': 'close-button', 'modal': 'RV-settings'})  # 'RV-close-settings-button')
                        ]
                    )
                ],
            ),

            ##### Save modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-save'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Save to'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Div([
                            html.Span('Save-file directory:'),
                            html.Span(f'{save_file_path}'),
                        ]),
                        html.Div([
                            html.Span('Save-file name:'),
                            dcc.Input(
                                id='RV-save-file-name',
                                type='text',
                                placeholder='Enter save-file name here...',
                                debounce=True
                            ),
                        ]),
                        # html.Div([
                        #     dcc.Dropdown(
                        #         id='RV-extension-selection-dropdown',
                        #         options=[{'label': '.fif', 'value': '.fif'}],#, {'label': '.edf', 'value': '.edf'}, {'label': '.set (EEGLAB)', 'value': '.set'}],
                        #         # placeholder='Select file extension (default .fif)',
                        #         value='.fif',
                        #     ),
                        # ]),
                        html.Div(className='aligned',
                            children=[
                                html.Button(
                                    'Save with entered name (.fif)',
                                    id='RV-save-button',
                                ),
                                html.Button(
                                    'Overwrite current file',
                                    id={'type': 'open-button', 'modal': 'RV-overwrite'},
                                    disabled=True
                                ),
                                html.Button(
                                    'Save annotations with entered name (.csv)',
                                    id='RV-save-annotations-button',
                                ),
                                html.Button(
                                    'Save bad channels with entered name (.txt)',
                                    id='RV-save-bad-channels-button',
                                ),
                            ]
                        ),
                    ]),
                    dbc.ModalFooter(
                        html.Button('Close', id={'type': 'close-button', 'modal': 'RV-save'}),
                    )
                ],
            ),

            ##### Overwrite modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-overwrite'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Caution!'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Span('Are you sure you want to overwrite:'),
                        html.Span(id='RV-overwrite-save-file')
                    ]),
                    dbc.ModalFooter(class_name='binary-modal-footer',
                        children=[
                            html.Button('Yes', id={'type': 'process-button', 'modal': 'RV-overwrite'}),  # 'RV-overwrite-button'
                            html.Button('Cancel', id={'type': 'close-button', 'modal': 'RV-overwrite'}),
                        ]
                    )
                ],
            ),

            ##### Remove-annotations modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-remove-annotation-label'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Caution!'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Div([
                            html.Span('Are you sure you want to remove this annotation label? All annotations with this label will be deleted.')
                        ])
                    ]),
                    dbc.ModalFooter(class_name='binary-modal-footer',
                        children=[
                            html.Button('Remove label', id={'type': 'process-button', 'modal': 'RV-remove-annotation-label'}),  # 'RV-remove-annotation-label-button'
                            html.Button('Cancel', id={'type': 'close-button', 'modal': 'RV-remove-annotation-label'}),
                        ]
                    )
                ],
            ),

            ##### Rename-annotations modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-rename-annotation-label'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Caution!'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Div([
                            html.Span('Enter the new annotation label below. All annotations with the current label will be renamed.')
                        ]),
                        html.Div([
                            dcc.Input(
                                id='RV-renamed-annotation-label',
                                placeholder='New annotation label',
                                debounce=True,
                                minLength=1
                            ),
                        ]),
                    ]),
                    dbc.ModalFooter(class_name='binary-modal-footer',
                        children=[
                            html.Button('Rename label', id={'type': 'process-button', 'modal': 'RV-rename-annotation-label'}),  # 'RV-rename-annotation-label-button'
                            html.Button('Cancel', id={'type': 'close-button', 'modal': 'RV-rename-annotation-label'}),
                        ]
                    )
                ],
            ),

            ##### Stats modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-stats'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Statistics'),
                        close_button=False,
                    ),
                    dbc.ModalBody(id='RV-stats-modal-body', children=[]),
                    dbc.ModalFooter(
                        html.Button('Close', id={'type': 'close-button', 'modal': 'RV-stats'}),
                    )
                ],
            ),

            ##### Power-spectrum modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-psd'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Power spectrum & Amplitude topomap'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Div([
                            html.H1('Selected interval:'),
                            html.Span(id='RV-selected-interval-psd'),
                        ]),
                        html.Div([
                            html.H1('Selected channels:'),
                            html.Span(id='RV-selected-channels-psd'),
                        ]),
                        html.Div([
                            html.H1('Flat channels:'),
                            html.Span(id='RV-flat-selected-channels-psd'),
                        ]),
                        html.Div([
                            html.H1('Welch power spectrum:'),
                            dcc.Loading(
                                id='RV-psd-graph-loading',
                                type='default',
                                # color='red',
                                children=[
                                    dcc.Graph(
                                        id='RV-psd-graph',
                                        config={
                                            'displaylogo': False,
                                            'displayModeBar': True,
                                            'modeBarButtonsToRemove':[
                                                'select2d',
                                                'lasso2d',
                                                'autoScale2d',
                                                'toImage'
                                            ],
                                            'scrollZoom': True,
                                        },
                                    ),
                                ]
                            )
                        ]),
                        html.Hr(),
                        html.Div([
                            html.H1('Amplitude topomap:'),
                            html.Div([
                                html.Span('Animation sampling rate (in Hz):'),
                                dcc.Input(
                                    id='RV-topomap-sampling-rate-input',
                                    className='medium-input',
                                    type='number',
                                    placeholder='all frames',
                                    min=1,
                                    step=1,
                                    value=10,
                                    debounce=True,
                                ),
                            ]),
                            html.Div([
                                dcc.Checklist(
                                    id='RV-topomap-bad-channel-interpolation',
                                    inline=True,
                                    options=[
                                        {'label': 'Interpolate bad channels', 'value': 1},
                                    ],
                                    value=[1]
                                )
                            ]),
                            html.Div([
                                html.Button(
                                    'Compute topomap animation',
                                    id='RV-topomap-button',
                                ),
                            ]),
                            html.Div([
                                dcc.Loading(
                                    id='RV-topomap-graph-loading',
                                    type='default',
                                    # color='red',
                                    children=[
                                        dcc.Graph(
                                            id='RV-topomap-graph',
                                            config={
                                                'displaylogo': False,
                                                'displayModeBar': True,
                                                'modeBarButtonsToRemove':[
                                                    'select2d',
                                                    'lasso2d',
                                                    'autoScale2d',
                                                    'toImage'
                                                ],
                                                'scrollZoom': True,
                                            },
                                            style={'display': 'none'},
                                        ),
                                    ]
                                )
                            ]),
                        ])
                    ]),
                    dbc.ModalFooter(
                        html.Button('Close', id={'type': 'close-button', 'modal': 'RV-psd'}),
                    )
                ],
            ),

            ##### Help modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-help'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Help'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        c.MANUAL_SUMMARY
                    ]),
                    dbc.ModalFooter(
                        html.Button('Close', id={'type': 'close-button', 'modal': 'RV-help'}),
                    )
                ],
            ),

            ##### Quit modal #####
            dbc.Modal(id={'type': 'modal', 'modal': 'RV-quit'},
                is_open=False,
                size='lg',
                centered=True,
                children=[
                    dbc.ModalHeader(
                        children=dbc.ModalTitle('Caution!'),
                        close_button=False,
                    ),
                    dbc.ModalBody([
                        html.Div([
                            html.Span('Are you sure you want to quit? ' + (f'Data will be saved to {temp_save_file_path}.' if auto_save else 'Make sure you have saved all relevant data.'))
                        ])
                    ]),
                    dbc.ModalFooter(class_name='binary-modal-footer',
                        children=[
                            html.Button('Yes', id={'type': 'process-button', 'modal': 'RV-quit'}),  # 'RV-final-quit-button'
                            html.Button('Cancel', id={'type': 'close-button', 'modal': 'RV-quit'}),
                        ]
                    )
                ],
            ),
        ]
    )

    @callback(
        Output({'type': 'modal', 'modal': MATCH}, 'is_open', allow_duplicate=True),
        [
            Input({'type': 'open-button', 'modal': MATCH}, 'n_clicks'),
            Input({'type': 'close-button', 'modal': MATCH}, 'n_clicks')
        ],
        State({'type': 'modal', 'modal': MATCH}, 'is_open'),
        prevent_initial_call=True
    )
    def toggle_modal(open_button, close_button, is_open):
        """Toggles modal connected to respective open/close buttons.
        """
        return not is_open

    @callback(
        Output({'type': 'modal', 'modal': MATCH}, 'is_open', allow_duplicate=True),
        Input({'type': 'process-button', 'modal': MATCH}, 'n_clicks'),
        State({'type': 'modal', 'modal': MATCH}, 'is_open'),
        prevent_initial_call=True
    )
    def toggle_modal(process_button, is_open):
        """Toggles modal connected to respective process button (if applicable).
        """
        return not is_open

    return layout
