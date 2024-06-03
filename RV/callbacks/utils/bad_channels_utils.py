from pyprep.find_noisy_channels import NoisyChannels

import mne


def bad_channel_disagrees(channel_name: str, bad_channels_dict: dict):
    """Checks if given channel_name is contained in all bad-channel sources (except ones that marked no bad channels) in bad_channels_dict.

    Args:
        channel_name (str): Name of channel to be checked.
        bad_channels_dict (dict): Dictionary with source of bad channels as key and bad channels as values.

    Returns:
        bool: Whether or not the given channel_name disagrees with any source.
    """
    bad_channel_disagrees = False

    for source, bad_channels in bad_channels_dict.items():
        # Ignore sources where no bad channels were marked
        if len(bad_channels) == 0:
            continue
        elif channel_name not in bad_channels:
            bad_channel_disagrees = True
            break
        else:
            continue

    return bad_channel_disagrees

def get_automatic_bad_channels(raw: mne.io.Raw, bad_channel_detection='RANSAC'):
    """Performs automatic bad-channel detection on mne.io.Raw object using method given by bad_channel_detection.

    Args:
        raw (mne.io.Raw): Raw object holding (preprocessed) EEG data.
        bad_channel_detection (str, optional): Name of desired bad-channel detection method. Defaults to 'RANSAC'.

    Raises:
        NotImplementedError: If given bad-channel detection method has not been implemented.

    Returns:
        list: List of strings of bad-channel names.
    """
    detected_bad_channels = []

    if bad_channel_detection == 'RANSAC':
        raw_copy = raw.copy()

        # Reset bad channels before automatic detection
        raw_copy.info['bads'] = []

        # Resample for faster execution (if raw_copy.info['sfreq'] > 250.0) and optimized parameters
        raw_copy.resample(250.0)

        nd = NoisyChannels(raw_copy, do_detrend=True, random_state=None, matlab_strict=False)
        nd.find_bad_by_deviation(deviation_threshold=4)
        nd.find_bad_by_ransac(
            n_samples=50,
            sample_prop=0.3,
            corr_thresh=0.7,
            frac_bad=0.4,
            corr_window_secs=5.0,
            channel_wise=False,
            max_chunk_size=None)
        nd.find_bad_by_SNR()

        detected_bad_channels = nd.bad_by_ransac + nd.bad_by_SNR + nd.bad_by_deviation
        detected_bad_channels = list(set(detected_bad_channels))

    # TODO: Implement other automatic bad-channel-detection methods here
    # elif bad_channel_detection == INSERT_YOUR_METHOD_HERE:

    else:
        raise NotImplementedError("Currently, only 'RANSAC' is implemented for automatic bad-channel detection.")

    return detected_bad_channels
