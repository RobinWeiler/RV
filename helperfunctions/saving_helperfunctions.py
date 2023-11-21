import os

import pandas as pd

from helperfunctions.annotation_helperfunctions import get_annotations_dict
from helperfunctions.stats_helperfunctions import _natural_keys

import constants as c


def save_to(save_file_name, extension, raw):
    """Saves given raw to given file-name with given extension. If Save-file-name already exists, integer is added to the end.

    Args:
        save_file_name (string): Desired name of save-file.
        extension (str): Desired save-file extension. Currently supported file-formats are .fif, .edf, .set.
        raw (mne.io.Raw): Raw object to save.

    Returns:
        string: Name of save-file.
    """
    new_file_name = save_file_name
    existing_file_counter = 0

    while os.path.exists(os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name + extension)):
        existing_file_counter += 1
        new_file_name = save_file_name + '-' + str(existing_file_counter)
        
        print('File already exists, trying: {}'.format(new_file_name))

    new_file_name = new_file_name + extension
    new_file_path = os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name)

    # print('Saving data to {}'.format(new_file_path))

    if extension == '.fif':
        print('Saving data to .fif file')
        raw.save(new_file_path, picks='all', overwrite=False)
    # elif extension == '.edf':
    #     print('Saving data to .edf file')
    #     raw.export(new_file_path, fmt='edf')
    # elif extension == '.set':
    #     print('Saving data to .set (EEGLAB) file')
    #     print(raw.info)
    #     print(raw.filenames[0])
    #     raw.export(new_file_path, fmt='eeglab')
    else:
        print('Extension not recognized')

    return new_file_name

def overwrite_save(loaded_file_name, raw, save_file_path=None):
    """Overwrites currently loaded file with given raw object. Only possible if currently loaded file is in "save-files" directory or at given save-file-path. If custom save_file_path is given, additionally saves a flag-file with same file-name and .done extension. This can be used by external scripts to detect when user is done annotating and has saved data.

    Args:
        loaded_file_name (string): File-name of currently loaded data.
        raw (mne.io.Raw): Raw object to save.
        save_file_path (string, optional): Custom path to save-file to overwrite. Defaults to None and overwrites save-file-name in "save-files" directory.
    """
    if save_file_path:
        print('Writing to {}'.format(save_file_path))
        raw.save(save_file_path, picks='all', overwrite=True)
        
        extension_index = save_file_path.rfind('.')
        flag_file_path = save_file_path[:extension_index] + '.done'
        
        f = open(flag_file_path, "x")
        f.close()
        
        return
    else:
        old_file_path = os.path.join(c.SAVE_FILES_DIRECTORY, loaded_file_name)

    print('Will overwrite {}'.format(old_file_path))

    if os.path.exists(old_file_path):
        print('Overwriting {}'.format(old_file_path))
        raw.save(old_file_path, picks='all', overwrite=True)
    else:
        print('Save-file to overwrite does not exist')  # Show alert
        
    return

def save_annotations(save_file_name, raw):
    """Saves annotations of given raw to given file-name in .csv file. If Save-file-name already exists, integer is added to the end.

    Args:
        save_file_name (string): Desired name of save-file.
        raw (mne.io.Raw): Raw object to save.
    """
    new_file_name = save_file_name
    existing_file_counter = 0

    while os.path.exists(os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name + '.csv')):
        existing_file_counter += 1
        new_file_name = save_file_name + '-' + str(existing_file_counter)
        
        print('File already exists, trying: {}'.format(new_file_name))

    new_file_name = new_file_name + '.csv'
    new_file_path = os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name)

    # print('Saving data to {}'.format(new_file_path))
    
    annotations_dict = get_annotations_dict(raw)
    df = pd.DataFrame(annotations_dict)

    print('Saving annotations to {}'.format(new_file_path))
    df.to_csv(new_file_path)

def save_bad_channels(save_file_name, raw):
    """Saves bad channels of given raw to given file-name in .txt file. If Save-file-name already exists, integer is added to the end.

    Args:
        save_file_name (string): Desired name of save-file.
        raw (mne.io.Raw): Raw object to save.
    """
    new_file_name = save_file_name
    existing_file_counter = 0

    while os.path.exists(os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name + '.txt')):
        existing_file_counter += 1
        new_file_name = save_file_name + '-' + str(existing_file_counter)
        
        print('File already exists, trying: {}'.format(new_file_name))

    new_file_name = new_file_name + '.txt'
    new_file_path = os.path.join(c.SAVE_FILES_DIRECTORY, new_file_name)

    # print('Saving data to {}'.format(new_file_path))
    
    bad_channels = raw.info['bads']
    bad_channels.sort(key=_natural_keys)

    print('Saving bad channels to {}'.format(new_file_path))
    
    textfile = open(new_file_path, "w")
    for channel in bad_channels:
        textfile.write(channel + ", ")
    textfile.close()

def quick_save(raw):
    """Overwrites automatic temorary save-file with given raw object.

    Args:
        raw (mne.io.Raw): Raw object to save.
    """
    if not raw:
        print('Error: No raw datastructure given!')
        return

    # print('Saving changes to {}'.format(temp_save_path))
    raw.save(c.TEMP_SAVE_PATH, picks='all', overwrite=True)
    return
