import mne

mne.set_log_level(verbose='INFO')


def filter_fir(Raw, hp, lp):
    """Returns bandpass-filtered with FIR EEG recording.

    Args:
        Raw (mne.io.Raw): EEG recording to be filtered.
        hp (float): High-pass cutoff frequency in Hz.
        lp (float): Low-pass cutoff frequency in Hz.

    Returns:
        Raw (mne.io.Raw): Filtered signal.
    """
    # Filter FIR
    if Raw.info['highpass'] != hp and Raw.info['lowpass'] != lp:
        Raw.filter(hp, lp, fir_design="firwin", verbose=0)
    return Raw


def interpolate_bads(Raw):
    """Returns EEG recording with interpolated bad channels.

    Args:
        Raw (mne.io.Raw): EEG recording for which bad channels will be interpolated.

    Returns:
        Raw (mne.io.Raw): EEG recording with interpolated bad channels.
    """
    # Interpolate bad channels
    if len(Raw.info['bads']) > 0:
        Raw.interpolate_bads(reset_bads=False)
    return Raw


def reref(Raw, reference):
    """Returns re-referenced EEG recording.

    Args:
        Raw (mne.io.Raw): EEG recording to be re-referenced.
        reference (str): New reference.

    Returns:
        Raw (mne.io.Raw): Re-referenced EEG recording.
    """
    # Re-reference
    if not Raw.info['custom_ref_applied']:
        Raw.set_eeg_reference(ref_channels=reference, projection=True, verbose=0)
    return Raw

def resample(Raw, sample_frequency):
    """Returns resampled EEG recording.

    Args:
        Raw (mne.io.Raw): EEG recording to be resampled.
        sample_frequency (float): New sampling frequency.

    Returns:
        Raw (mne.io.Raw): Resampled EEG recording.
    """
    # Resample
    if Raw.info['sfreq'] != sample_frequency:
        Raw.resample(sample_frequency)
    return Raw