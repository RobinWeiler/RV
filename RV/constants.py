from collections import OrderedDict

from dash import dcc, html


##### Conversion variables #####
# CONVERSION_VALUE_MICROVOLTS_TO_VOLTS = 1e-6
CONVERSION_VALUE_VOLTS_TO_MICROVOLTS = 1e+6

##### Default plotting variables #####
DEFAULT_Y_AXIS_OFFSET = 40
DEFAULT_SEGMENT_SIZE = 10

##### Default colors #####
BACKGROUND_COLOR = '#dfdfdf'
PLOT_COLOR = '#fafafa'
PRIMARY_BLUE = '#2bb1d6'
BAD_CHANNEL_COLOR = '#eb4757'
BAD_CHANNEL_DISAGREE_COLOR = 'sandybrown'
ANNOTATION_COLOR_OPTIONS = ['red', 'blue', 'yellow', 'green', 'purple', 'turquoise']
MODEL_CHANNEL_COLOR = PRIMARY_BLUE  # '#4DAA57'

##### Mapping of EGI-129 channels to brain lobes #####
CHANNELS_TO_LOBES_EGI129 = {
    'Frontal': ['E1', 'E2', 'E3', 'E4', 'E8', 'E9', 'E10', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E32', 'E33', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128'],
    'Temporal (l)': ['E28', 'E34', 'E35', 'E38', 'E39', 'E40', 'E41', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 'E56', 'E57'],
    'Central': ['E5', 'E6', 'E7', 'E11', 'E12', 'E13', 'E20', 'E29', 'E30', 'E31', 'E36', 'E37', 'E42', 'E54', 'E55', 'E79', 'E80', 'E87', 'E93', 'E104', 'E105', 'E106', 'E111', 'E112', 'E118'],
    'Temporal (r)': ['E97', 'E98', 'E100', 'E101', 'E102', 'E103', 'E107', 'E108', 'E109', 'E110', 'E113', 'E114', 'E115', 'E116', 'E117', 'E119', 'E120', 'E121'],
    'Parietal': ['E52', 'E53', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E72', 'E77', 'E78', 'E84', 'E85', 'E86', 'E90', 'E91', 'E92', 'E94', 'E95', 'E96', 'E99'],
    'Occipital': ['E69', 'E70', 'E71', 'E73', 'E74', 'E75', 'E76', 'E81', 'E82', 'E83', 'E88', 'E89'],
    'Reference': ['Cz']
}
CHANNELS_TO_LOBES_EGI129 = OrderedDict(CHANNELS_TO_LOBES_EGI129)
EGI129_CHANNELS = [channel_name for channel_names in CHANNELS_TO_LOBES_EGI129.values() for channel_name in channel_names]

##### 10-20 system channel names #####
STANDARD_10_20 = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F8', 'T4', 'T6', 'F4', 'C4', 'P4', 'O2', 'Fz', 'Cz', 'Pz']
BIOSEMI64_10_20 = ['Fp1', 'F7', 'T7', 'P7', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F8', 'T8', 'P8', 'F4', 'C4', 'P4', 'O2', 'Fz', 'Cz', 'Pz']
TUAR_CHANNELS = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
EGI128_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124', 'E104', 'E92', 'E83', 'E11', 'Cz', 'E62']
EGI129_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124', 'E104', 'E92', 'E83', 'E11', 'E129', 'E62']
EGI128_2_10_20 = ['E22', 'E33', 'E45', 'E58', 'E24', 'E36', 'E52', 'E70', 'E9', 'E122', 'E108', 'E96', 'E124', 'E104', 'E92', 'E83', 'E11', 'E55', 'E62']
ADJACENT_10_20 = ['E18','E27','E46','E59','E27','E30','E51','E71','E15','E123','E100','E91','E4','E103','E86','E84','E16','E55','E68']

##### User manual plotted in help modal #####
MANUAL_SUMMARY = html.Div(id='RV-manual-summary-container',
    children=[
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
            dcc.Markdown('''
                # Keyboard shortcuts

                - **a**: Activate annotation-marking mode (button 3 in menu bar)
                - **delete**/**backspace**: Delete currently selected annotation (button 4 in menu bar)
                - **b**: Activate bad-channel-marking mode (button 5 in menu bar)
                - **s**: Activate segment-selection mode to generate power spectrum (button 7 in menu bar)
                - **left arrow**: Plot previous segment (**←** button)
                - **right arrow**: Plot following segment (**→** button)
                - **up arrow**: Increase channel offset by 10 μV
                - **down arrow**: Decrease channel offset by 10 μV
                - **+**: Increase scale by 0.5
                - **-**: Decrease scale by 0.5
            '''),

            html.H1('User Interface Overview'),
            html.Div([
                html.Img(src='assets/RV_GUI_figure.png', alt='Annotated GUI', style={'max-width': '100%'})
            ]),
            dcc.Markdown('''
                The main graphical user interface (GUI) of Robin's Viewer (RV) has four subsections (highlighted in image).

                ## Menu bar

                From left to right, there is the following functionality:

                1. **Settings**: Opens "Settings" pop-up window you are presented with when starting RV.
                2. **Save to**: Opens a pop-up window with several options to save the (annotated) EEG data.
                3. **Annotate segment**: While this button is active (outlined in blue), you can select a segment of the plot to annotate (by clicking and dragging on the plot).
                    This creates a semi-transparent box (colored according to the Annotation settings in the "Settings" pop-up window) spanning the entire vertical axis in view across the selected time interval.
                    These annotations can be freely adjusted in their position and size, or removed entirely (with button 4), when clicked on their edges.
                    The annotation is saved with the label set in the Annotation settings in the "Settings" pop-up window.
                    This button is activated by default.
                4. **Delete annotation**: Removes the currently selected (by clicking on its edges) annotation.
                5. **Mark bad channels**: While this button is active (outlined in blue), you can click on a trace to mark it as a bad channel.
                    The respective trace will be marked in red (or yellow if this marking disagrees with previous markings; see **Stats**).
                    Clicking on a bad channel again will remove the marking.
                6. **Hide bad channels**: Hides all marked bad channels. Clicking a second time shows all channels again.
                7. **Select segment**: While this button is active (outlined in blue), you can select a segment of the plot to generate a power spectrum of (by clicking and dragging).
                8. **Power spectrum**: Opens a pop-up window displaying the Welch power spectrum of the currently selected interval of data.
                    Here, you can also choose to compute an animated amplitude-topomap of the selected time interval (across all channels).
                9. **Pan**: While this button is active (outlined in blue), you can move the currently plotted segment (by clicking and dragging).
                10. **Zoom**: While this button is active (outlined in blue), you can select a segment of the plot to zoom-in on (by clicking and dragging).
                11. **Reset view**: Resets the axes of the currently plotted segment. Use this after zooming.
                12. **Stats**: Opens a pop-up window displaying statistics surrounding the annotated data. Currently, the implemented statistics are:
                    - the recording's total length (in seconds) 
                    - the amount of non-annotated data (in seconds)
                    - the amount of consecutive, non-annotated intervals longer than 2 seconds 
                    - a histogram to show the distribution of the length of non-annotated data segments
                    - the amount of annotated data (in seconds) in total and for each annotation label
                    - the percentage of overlap between each annotation label (if applicable) measured in 100 millisecond windows
                    - marked bad channels and at what stage they were marked (e.g., current session, in the loaded file, in loaded bad-channel files)
                    - the bad channels that are not included in all bad-channel sources (and hence "disagree")
                13. **Help**: Opens this pop-up window.
                14. **Quit**: Opens a pop-up window asking for confirmation to quit RV.
                    After confirming, plot is reset.
                    Before quitting RV, you should save the annotated data using the “Save to” button (2.) mentioned above (if you forget to save the data, it can be restored by loading the "Temporary save file" immediately after RV is restarted).
                    You do not have to manually save the data if auto_save is set to True.

                ## View-slider

                The view-slider is located at the bottom of the screen.
                It can be used to switch between plotted segments (if applicable) by clicking on the blue dots or the numbers indicating the beginning of the respective segment (in seconds).
                The segment size, by default initialized to 10 seconds, can be adjusted in the "Visualization settings" of the "Settings" pop-up window.
                The **←** and **→** buttons plot the previous and following segments, respectively.
                The view-slider also provides an overview over the annotations made.

                ## Legend

                At the right side of the plot, there is a scrollable legend showing the names of all plotted EEG channels.
                Clicking on any channel name hides the channel from the plot.
                Double-clicking a channel name hides all channels except for the selected one.
                You can follow this up by adding more channels to be displayed by clicking their names once.
                Isolating specific channels can be useful when selecting an interval of which to calculate the power spectrum using button 7 of the menu bar.
                The 

                ## Plot

                The majority of the screen is occupied by the main plot which dynamically adapts its size based on RV's window size to fill as much space as possible.
                All selected channels are spread across the vertical axis with an offset of 40 μV between them by default, unless specified otherwise in the "Visualization settings" of the "Settings" pop-up window.
                Time, on the horizontal axis, is displayed in seconds. 
                The EEG traces are plotted in black, with the exception of bad channels, whose traces are marked in red (or yellow if this marking disagrees with previous markings; see button 12 (**Stats**) of menu bar).
                You can hover your mouse on the plot in order to display the time (in seconds) and amplitude (in μV; rounded to two decimals) of the trace hovered on.
                If deep-learning predictions are used, they are plotted below the EEG traces with a colorscale ranging from red to white to blue indicating confidence scores of 1, 0.5, and 0, respectively.
                Clicking and dragging on either axis allows to move the plotted segment.

            ''')
        ])
    ]
)
