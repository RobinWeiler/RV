import multiprocessing

import mne


def preprocess_EEG(raw: mne.io.Raw, high_pass=None, low_pass=None, notch_freq=None, reference=None):
    """Filter and re-reference given mne.io.Raw object based on input values using mne.

    Args:
        raw (mne.io.Raw): Raw object holding EEG data.
        high_pass (float, optional): High-pass value for bandpass filter. Defaults to None.
        low_pass (float, optional): Low-pass value for bandpass filter. Defaults to None.
        notch_freq (float, optional): Frequency of notch filter. Defaults to None.
        reference (str, optional): One of 'None', 'average' or a channel name (e.g., 'Cz') to re-reference data to. Defaults to None.

    Returns:
        mne.io.Raw: Preprocessed mne.io.Raw object.
    """
    num_cores = multiprocessing.cpu_count()
    if high_pass != raw.info['highpass'] or low_pass != raw.info['lowpass']:
        raw.filter(high_pass, low_pass, fir_window='hamming', fir_design='firwin', phase='zero', pad='reflect_limited', verbose=0, n_jobs=num_cores)

    if notch_freq:
        raw.notch_filter(notch_freq, method='fir', verbose=0, n_jobs=num_cores)

    if reference == 'None' or reference is None:
        print('No re-referencing')
        reference = None
    elif reference == 'average':
        pass
    else:
        reference = [reference]

    if reference:
        print('Applying custom reference {}'.format(reference))
        raw.set_eeg_reference(reference, verbose=0)

    return raw
