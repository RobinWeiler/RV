import mne

mne.set_log_level(verbose='INFO')


def filter_fir(raw, hp, lp):
    """Returns bandpass-filtered with FIR EEG recording.

    Args:
        raw (mne.io.raw): EEG recording to be filtered.
        hp (float): High-pass cutoff frequency in Hz.
        lp (float): Low-pass cutoff frequency in Hz.

    Returns:
        raw (mne.io.raw): Filtered signal.
    """
    # Filter FIR
    if raw.info['highpass'] != hp and raw.info['lowpass'] != lp:
        raw.filter(hp, lp, fir_design="firwin", verbose=0)
    return raw


def interpolate_bads(raw):
    """Returns EEG recording with interpolated bad channels.

    Args:
        raw (mne.io.raw): EEG recording for which bad channels will be interpolated.

    Returns:
        raw (mne.io.raw): EEG recording with interpolated bad channels.
    """
    # Interpolate bad channels
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=False)
    return raw


def reref(raw, reference):
    """Returns re-referenced EEG recording.

    Args:
        raw (mne.io.raw): EEG recording to be re-referenced.
        reference (str): New reference.

    Returns:
        raw (mne.io.raw): Re-referenced EEG recording.
    """
    # Re-reference
    if not raw.info['custom_ref_applied']:
        raw.set_eeg_reference(ref_channels=reference, projection=False, verbose=0)
    return raw

def resample(raw, sample_frequency):
    """Returns resampled EEG recording.

    Args:
        raw (mne.io.raw): EEG recording to be resampled.
        sample_frequency (float): New sampling frequency.

    Returns:
        raw (mne.io.raw): Resampled EEG recording.
    """
    # Resample
    if raw.info['sfreq'] != sample_frequency:
        raw.resample(sample_frequency)
    return raw