import os
import numpy as np

# Default conversion variables
CONVERSION_VALUE_MICROVOLTS_TO_VOLTS = 1e-6
CONVERSION_VALUE_VOLTS_TO_MICROVOLTS = 1e+6

DEFAULT_Y_AXIS_OFFSET = 40
DEFAULT_SEGMENT_SIZE = 10

BAD_CHANNEL_COLOR = '#eb5f6e'  # '#8f8f8f'
BAD_CHANNEL_DISAGREE_COLOR = 'orange'

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
    'frontal': [1, 2, 3, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 32, 33, 122, 123, 124, 125, 126, 127, 128],
    'temporal_left': [28, 34, 35, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 56, 57],
    'central': [5, 6, 7, 11, 12, 13, 20, 29, 30, 31, 36, 37, 42, 54, 55, 79, 80, 87, 93, 104, 105, 106, 111, 112, 118],
    'temporal_right': [97, 98, 100, 101, 102, 103, 107, 108, 109, 110, 113, 114, 115, 116, 117, 119, 120, 121],
    'parietal': [52, 53, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 77, 78, 84, 85, 86, 90, 91, 92, 94, 95, 96, 99],
    'occipital': [69, 70, 71, 73, 74, 75, 76, 81, 82, 83, 88, 89],
}