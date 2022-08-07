
import mne

mne.set_log_level(verbose='INFO')


def filter_fir(Raw, hp, lp):
    # Filter FIR
    if Raw.info['highpass'] != hp and Raw.info['lowpass'] != lp:
        Raw.filter(hp, lp, fir_design="firwin", verbose=0)
    return Raw


def interpolate_bads(Raw):
    # Interpolate bad channels
    if len(Raw.info['bads']) > 0:
        Raw.interpolate_bads(reset_bads=False)
    return Raw


def reref(Raw, reference):
    # Re-reference
    if not Raw.info['custom_ref_applied']:
        Raw.set_eeg_reference(ref_channels=reference, projection=True, verbose=0)
    return Raw

def resample(Raw, sample_frequency):
    # Resample
    if Raw.info['sfreq'] != sample_frequency:
        Raw.resample(sample_frequency)
    return Raw