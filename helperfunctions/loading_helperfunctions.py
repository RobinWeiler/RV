import os

import pandas as pd
import numpy as np
import mne

import constants as c


def parse_data_file(filename):
    """Loads data of given file-name into mne.io.Raw object. Currently supported file-formats are .fif, .edf, .bdf, .set, and .csv. This is also used to load automatic temporary save-file.

    Args:
        filename (string): File-name to load. Has to be located in 'data' or 'save_files' directory.

    Returns:
        mne.io.Raw: Loaded raw object.
    """
    print(filename)

    data_file = os.path.join(c.DATA_DIRECTORY, filename)
    save_file = os.path.join(c.SAVE_FILES_DIRECTORY, filename)

    if os.path.exists(data_file):
        print('Found data-file')
        file = data_file
    elif os.path.exists(save_file):
        print('Found save-file')
        file = save_file
    # Always loads automatic temporary save-file if file-name not found
    elif os.path.exists(c.TEMP_SAVE_PATH):
        file = c.TEMP_SAVE_PATH
        print('Restored last working save-file')
    else:
        print('Error: Could not find file. Make sure file is located in {} or {} directory.'.format(c.DATA_DIRECTORY, c.SAVE_FILES_DIRECTORY))

    # try:
    if '.csv' in filename:  # For backwards-compatibility
        df = pd.read_csv(file)

        EEG_data = df.values
        # print(EEG_data.shape)

        # Extract time-scale
        timestep = EEG_data[1, 0]  # First value after 0 in time-column
        # print("One time step = {} milliseconds".format(timestep))

        timestep = float(timestep) / 1000 # Convert to seconds
        # print('One time step = {} seconds'.format(timestep))

        global sample_rate
        sample_rate = 1 / timestep
        # print('Sampling rate = {} Hz'.format(sample_rate))

        # Remove time-column from data
        EEG_data = EEG_data[:,1:]

        # Convert microvolts to volts for backwards compatibility
        EEG_data = EEG_data * c.CONVERSION_VALUE_MICROVOLTS_TO_VOLTS

        # Extract channel names
        column_names = []
        for col in df.columns:
            column_names.append(col)
            
        # For debugging
        # print(column_names)
            
        # Remove "Time" from the channel names    
        column_names.pop(0)

        info = mne.create_info(ch_names=column_names, sfreq=sample_rate, ch_types='eeg')
        raw = mne.io.RawArray(np.transpose(EEG_data), info)
        print(raw.info)

        return raw
    elif '.edf' in filename:
        raw = mne.io.read_raw_edf(file, stim_channel=False, preload=True, verbose=True)
        print(raw.info)

        return raw
    elif '.bdf' in filename:
        raw = mne.io.read_raw_bdf(file, stim_channel=False, preload=True, verbose=True)
        print(raw.info)

        return raw
    elif '.set' in filename:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=True)
        print(raw.info)

        return raw
    elif '.fif' in filename:
        raw = mne.io.Raw(file, preload=True, verbose=True)
        print(raw.info)

        return raw
    else:
        print('This file type is not supported yet!')

def parse_model_output_file(filename, raw=None):
    """Loads model-output from given file-name. Currently supported file-formats are .csv, .txt, and .npy.

    Args:
        filename (string): Model-output file-name to load. Has to be located in 'data' directory.
        raw (mne.io.Raw): Raw object to retrieve annotations from. Defaults to None.

    Returns:
        tuple(np.array, list, float): One-dimensional array holding model-output, list of channel names used, and sample frequency. The latter allows for custom scaling of predictions.
    """
    data_file = os.path.join(c.DATA_DIRECTORY, filename)
    save_file = os.path.join(c.SAVE_FILES_DIRECTORY, filename)

    if os.path.exists(data_file):
        print('Found data-file')
        file = data_file
    elif os.path.exists(save_file):
        print('Found save-file')
        file = save_file
    else:
        print('Error: Could not find file. Make sure file is located in {} or {} directory.'.format(c.DATA_DIRECTORY, c.SAVE_FILES_DIRECTORY))

    if raw:
        if '.csv' in filename:
            df = pd.read_csv(file)

            sampling_frequency = raw.info['sfreq']
            print(sampling_frequency)
            timestep = 1 / sampling_frequency

            annotation_onsets = df['onset'].tolist()
            annotation_durations = df['duration'].tolist()
            annotation_ends = [x + y for x, y in zip(annotation_onsets, annotation_durations)]

            model_output = np.zeros(raw.__len__())

            for index, annotation_start in enumerate(annotation_onsets):
                annotation_start_index = int(annotation_start * sampling_frequency)
                annotation_end_index = int(annotation_ends[index] * sampling_frequency)

                for timepoint in np.arange(start=annotation_start_index, stop=annotation_end_index, step=1):
                    model_output[timepoint] = 1

            return model_output, None, sampling_frequency
        elif '.txt' in filename:
            model_output = np.loadtxt(file)
            assert model_output.shape[0] == raw.__len__(), 'Loaded predictions do not contain 1 prediction per timepoint in the raw EEG data.'

            return model_output, None, None
        elif '.npy' in filename:
            model_output = np.load(file)
            assert model_output.shape[0] == raw.__len__(), 'Loaded predictions do not contain 1 prediction per timepoint in the raw EEG data.'

            return model_output, None, None
        else:
            print('Wrong file type!')
    else:
        print('Make sure to load the accompanying EEG data first')
        return None, None, None