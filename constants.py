import os
import numpy as np
from dash import dcc, html

# Default conversion variables
CONVERSION_VALUE_MICROVOLTS_TO_VOLTS = 1e-6
CONVERSION_VALUE_VOLTS_TO_MICROVOLTS = 1e+6

DEFAULT_Y_AXIS_OFFSET = 40
DEFAULT_SEGMENT_SIZE = 10

BAD_CHANNEL_COLOR = '#eb5f6e'
BAD_CHANNEL_DISAGREE_COLOR = '#FFC145'
MODEL_CHANNEL_COLOR = '#9046CF'

# Path variables
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(CURRENT_PATH, 'data')
SAVE_FILES_DIRECTORY = os.path.join(CURRENT_PATH, 'save_files')
TEMP_SAVE_PATH = os.path.join(CURRENT_PATH, 'temp_raw.fif')
ASSETS_DIRECTORY = os.path.join(CURRENT_PATH, 'assets')
MODEL_DIRECTORY = os.path.join(CURRENT_PATH, 'model')

TITLE_IMAGE_FILE = os.path.join(ASSETS_DIRECTORY, 'title_image.png')

# Automatic bad-channel detection parameters
WINDOW_SIZE = 1.0
WINDOW_OVERLAP = 0.5
# Autoreject parameters (n_interpolate - number of channels to interpolate; consensus - proportion of bad sensors to call epoch as bad)
N_INTERPOLATE = np.array([1, 4, 32])
CONSENSUS = np.linspace(0, 1, 11)
THRESHOLD = 0.8  # ratio of segments (when >= THRESHOLD, mark as bad channel)

CHANNEL_TO_REGION_128 = {
    'Frontal': [1, 2, 3, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 32, 33, 122, 123, 124, 125, 126, 127, 128],
    'Temporal (left)': [28, 34, 35, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 56, 57],
    'Central': [5, 6, 7, 11, 12, 13, 20, 29, 30, 31, 36, 37, 42, 54, 55, 79, 80, 87, 93, 104, 105, 106, 111, 112, 118],
    'Temporal (right)': [97, 98, 100, 101, 102, 103, 107, 108, 109, 110, 113, 114, 115, 116, 117, 119, 120, 121],
    'Parietal': [52, 53, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 77, 78, 84, 85, 86, 90, 91, 92, 94, 95, 96, 99],
    'Occipital': [69, 70, 71, 73, 74, 75, 76, 81, 82, 83, 88, 89],
}

HELP_MENU = html.Div([
                html.Div([
                    html.H1('Relevant Links'),
                    html.Div([
                        html.Label([
                            'If you use this software for publications, please cite this paper: ',
                            html.Blockquote('Weiler, R., Diachenko, M., Juarez-Martinez, E.L., Avramiea, A.E., Bloem, P. and Linkenkaer-Hansen, K., 2023. Robin’s viewer: using deep-learning predictions to assist EEG annotation. Frontiers in Neuroinformatics, 16, p.1025847.'),
                        ]),
                    ]),
                    html.Div([
                        html.Label([
                            'Paper: ',
                            html.A(
                                'https://doi.org/10.3389/fninf.2022.1025847',
                                href='https://doi.org/10.3389/fninf.2022.1025847',
                                target='_blank'
                            )
                        ]),
                    ]),
                    html.Div([
                        html.Label([
                            'GitHub: ',
                            html.A(
                                'https://github.com/RobinWeiler/RV',
                                href='https://github.com/RobinWeiler/RV',
                                target='_blank'
                            )
                        ]),
                    ]),
                    html.Div([
                        html.Label([
                            'Documentation: ',
                            html.A(
                                'https://robinweiler.github.io/RV/RV.html',
                                href='https://robinweiler.github.io/RV/RV.html',
                                target='_blank'
                            )
                        ])
                    ]),
                ]),
                html.Div([
                    html.H1('User Interface Overview'),
                    html.Div([
                        html.Img(src='assets/GUI_figure.png', alt='Annotated GUI', style={'max-width': '100%'})
                    ]),
                    dcc.Markdown('''
                        The main graphical user interface (GUI) of Robin’s Viewer (RV) has six subsections (highlighted in Figure A). Figure B shows a close-up of the taskbar highlighted in turquoise in Figure A.

                        ## Menu bar

                        From left to right, there is the following functionality:

                        1. **Preprocessing**: Opens “Preprocessing” pop-up window to load files, change preprocessing and visualization settings, and activate the integrated deep-learning model.
                        2. **Save to**: Opens a pop-up window with several options to save the (annotated) EEG data. 
                        3. **Highlight model channels**: Allows to highlight the channels used by the model to generate its predictions in purple.
                        4. **Rerun model**: Allows to re-calculate the predictions of the integrated deep-learning model. Becomes active when bad channels are changed, after which the model should be re-run as it interpolates bad channels for its predictions.
                        5. **Model threshold**: Used when “Annotate according to model(s)” is active in the Preprocessing window and at least one model (integrated or external predictions) is loaded. Predictions above the given threshold are annotated. The label for these annotations is “bad_artifact_model” by default.
                        6. **Hide/show bad channels**: Allows to hide all marked bad channels. Clicking a second time shows all channels again.
                        7. **Annotation settings**: Opens a pop-up window allowing to add labels for annotation drawing and change the color of annotations corresponding to each label. Removing an annotation label results in all corresponding annotations being removed.
                        8. **Stats**: Opens a pop-up window displaying statistics surrounding the annotated data. Currently, the implemented statistics are:
                            - the recording’s total length (in seconds) 
                            - the amount of clean data (in seconds)
                            - the amount of consecutive, clean intervals longer than 2 seconds 
                            - a histogram to show the distribution of the length of clean data segments
                            - the amount of annotated data (in seconds) in total and for each annotation label
                            - the percentage of overlap between each annotation label (measured in 100 millisecond windows)
                            - marked bad channels and where they were marked (e.g., current session, loaded raw file, loaded bad-channel files
                            - the bad channels that are not included in all loaded bad-channel sources (i.e., that disagree)
                        9. **Power spectrum**: Opens a pop-up window displaying the most prominent frequency, as well as the respective power spectrum for the currently selected time interval of EEG data. For the power-spectrum computation, the Welch method is used as implemented in the Python-library SciPy.
                        10. **Help**: Opens a pop-up window containing links to the documentation and GitHub.
                        11. **Quit**: Opens a pop-up window asking for confirmation to shut down RV. After confirming, shuts down the local server RV is running on. After this the RV interface cannot be used anymore. Before shutting down RV, you should save the annotated data using the “Save to” button (2.) mentioned above (if you forget to save the data, it can be restored using the “temp_raw.fif” file after RV is restarted).
                        12. **<-**: Left-arrow button plots the previous segment.
                        13. **-10s**: Moves the plotted view 10 seconds to the left. This button should only be used if the plotted segment length is bigger than 10 seconds.
                        14. **Segment slider**: The segment slider shows all segments the EEG recording is divided into based on the segment length used. Clicking on any segment will plot the respective segment.
                        15. **+10s**: Moves the plotted view 10 seconds to the right. This button should only be used if the plotted segment length is bigger than 10 seconds.
                        16. **->**: Right-arrow button plots the next segment.

                        ## Taskbar

                        From left to right, there is the following functionality:

                        1. Take a picture of the (annotated) EEG signal and download it as a .png file.
                        2. While this button is active, select an area to zoom in on (click-and-drag).
                        3. While this button is active, move view along the channel- or time-axis, or both simultaneously (click-and-drag).
                        4. While this button is active, click on a channel to mark it as a bad channel. After a brief loading period, the respective channel will be marked in gray. Clicking on a bad channel again will remove the marking. Also, while this button is active, select a segment of the data for which to calculate and display the main underlying frequency, as well as a power spectrum in the “Power-spectrum” pop-up window, which can be opened using the “Power spectrum” button in the menu bar (click-and-drag). It is possible to select all channels or only a few desired ones, as explained in Section “Legend.”
                        5. While this button is active, select a segment of the plot to annotate, which creates a semi-transparent red box (indicating the presence of an artifact) spanning the entire vertical axis in view (currently only annotations across all channels are supported) across the selected time interval (click-and-drag). These annotations can be freely adjusted in their position and size, or removed entirely (with button 6), when clicked on. The annotation is saved with “bad_artifact” as its description (see Section “Annotations” for more details). This tool is activated by default.
                        6. Delete the currently selected annotation.
                        7. Zoom in one step.
                        8. Zoom out one step.
                        9. Zoom out as much as necessary to show all channels and annotations for the entire duration of the recording (or segment), including all peaks in the data (potentially produced by artifacts).
                        10. Display rulers for both axes corresponding to the datapoint currently hovered on.

                        ## Labeled buttons

                        From left to right, there is the following functionality:
                        1. **Reset channel-axis**: Reset the current view along the y-axis.
                        2. **Reset time-axis**: Reset the plotted timeframe to the initial view.
                        
                        ## View-slider

                        The view-slider is turned off by default. If activated in the visualization settings of the “Preprocessing” pop-up window, it is located at the bottom of the screen. The view-slider can be used to continuously scroll through the recording (horizontally along the time-axis) by clicking and dragging it. The slider’s range, by default initialized to 10 seconds, can be adjusted by clicking and dragging the small rectangles at its edges. In this way, you can visualize anything from the entire recording (or segment) at once, down to several milliseconds. 
                        The same functionality as the view-slider can also be achieved by clicking and dragging on the time-axis’ labels or selecting the third button in the taskbar and then clicking and dragging on the plot directly.

                        ## Legend

                        At the right side of the plot, there is a scrollable legend showing the names of all plotted EEG channels. Clicking on any channel name hides the channel from the plot. Double-clicking a channel name hides all channels except for the selected one. You can follow this up by adding more channels to be displayed by clicking their names once. 
                        Isolating specific channels can be used to retrieve the most prominent underlying frequency and power spectrum of an interval using button 4 of the taskbar.

                        ## Plot

                        The majority of the screen is occupied by the plot which dynamically adapts its size based on RV’s window size to fill as much space as possible. All selected channels are spread across the vertical axis with an offset of 40 μV between them by default, unless specified otherwise in the visualization settings of the “Preprocessing” pop-up window. Time, on the horizontal axis, is displayed in seconds. 
                        The EEG traces are plotted in black, with the exception of bad channels, whose traces are shown in red. EOG channels are plotted in blue. You can hover over any given point in the plot in order to display the time (in seconds) and amplitude (in μV if no custom scaling was used) values, rounded to three decimals, of the trace under the mouse. If deep-learning predictions are activated in the “Preprocessing” pop-up window, they are plotted below the EEG traces.

                    ''')
                ])
            ])
