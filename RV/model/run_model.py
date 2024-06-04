import torch

import mne

from RV.model.model import load_model, model_predict
from RV.model.utils.preprocess import preprocess_data


DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_model(raw: mne.io.Raw):
    """Runs CNN model on time-frequency-transformed EEG data, trained to detect artifacts in EEG data.
    It returns predictions from 0 to 1, where 1 represents high confidence in an artifact being present and 0 represents high confidence in respective datapoints being clean.
    For more details, see "Diachenko, Marina, et al. 'Improved manual annotation of EEG signals through convolutional neural network guidance.' Eneuro 9.5 (2022)".

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.

    Returns:
        array, list: Array of model predictions and list of strings of channel names considered by the model.
    """
    TF_data, segments, selected_channel_names = preprocess_data(raw, DEVICE)

    model = load_model(DEVICE)

    model_output = model_predict(TF_data, segments, model, DEVICE)

    return model_output, selected_channel_names
