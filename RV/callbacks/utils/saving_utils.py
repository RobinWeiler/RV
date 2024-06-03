import mne
import pandas as pd


def save_annotations_csv(raw: mne.io.Raw, save_file_path: str):
    """Saves annotations of given mne.io.Raw object to save_file_path as .csv file.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        save_file_path (str): Path at which to save annotations file.
    """
    annotations_dict = {
        'onset': raw.annotations.onset,
        'duration': raw.annotations.duration,
        'description': raw.annotations.description
    }

    annotations_df = pd.DataFrame(annotations_dict)
    annotations_df.to_csv(save_file_path)

    print('Saved annotations to {}'.format(save_file_path))

    return

def save_bad_channels_txt(raw: mne.io.Raw, save_file_path: str):
    """Saves bad channels of given mne.io.Raw object to save_file_path as .txt file.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        save_file_path (str): Path at which to save bad-channels file.
    """
    bad_channels = raw.info['bads']
                    
    textfile = open(save_file_path, 'w')
    for channel in bad_channels:
        textfile.write(channel + ', ')
    textfile.close()

    print('Saved bad channels to {}'.format(save_file_path))

    return
