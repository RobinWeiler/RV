import os

from mne.io import read_raw, read_raw_egi


def get_file_selection_options(file_paths: dict, allowed_extensions=[]):
    """Generates list with file-selection options for dcc.Dropdown.

    Args:
        file_paths (dict): Dictionary with keys pointing to directories whose files are included in options.
        allowed_extensions (list, optional): List of strings of file extensions included in options. Defaults to [] (all extensions).

    Returns:
        list: List of dictionaries with keys 'label' and 'value'.
    """
    options = []

    for key, path in file_paths.items():
        if key == 'load_file_path' or key == 'save_file_path':
            # when external data is loaded and file selection is disabled anyway
            if not os.path.isdir(path):
                continue

            files = os.listdir(path)

            for file in files:
                if file == 'info.txt':
                    continue

                base_name, extension = os.path.splitext(file)

                if (extension in allowed_extensions) or (len(allowed_extensions) == 0):
                    options.append(
                        {
                            'label': f"{file} (from {os.path.basename(path)})",
                            'value': os.path.join(path, file)
                        }
                    )
        elif key == 'temp_save_file_path':
            if ('.fif' in allowed_extensions) or (len(allowed_extensions) == 0):
                options.append(
                    {
                        'label': 'Temporary save file',
                        'value': path
                    }
                )
        else:
            continue

    return options

def load_raw(file_path: str):
    """Loads EEG data from given file_path.

    Args:
        filepath (_type_): _description_

    Returns:
        mne.io.Raw: Raw object holding EEG data.
    """
    base_name, extension = os.path.splitext(file_path)

    if extension == '.raw':
        raw = read_raw_egi(file_path, preload=True, verbose=True)
    else:
        raw = read_raw(file_path, preload=True, verbose=True)

    return raw
