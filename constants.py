import os
import numpy as np

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
