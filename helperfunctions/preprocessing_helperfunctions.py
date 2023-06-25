from helperfunctions.bad_channel_helperfunctions import get_bad_channels

import globals

def preprocess_EEG(raw, high_pass, low_pass, reference, bad_channel_detection, bad_channel_interpolation):
    # Bandpass-filter
    if (high_pass or low_pass) and not (float(high_pass) == globals.raw.info['highpass'] and float(low_pass) == globals.raw.info['lowpass']):
        # print(high_pass)
        # print(low_pass)
        print('Applying bandpass-filter')
        raw.filter(high_pass, low_pass, method='fir', fir_window='blackman')

    print(raw.info['bads'])

    # Bad-channel detection
    if bad_channel_detection == 'None':
        print('No automatic bad-channel detection')
        bad_channel_detection = None
    elif bad_channel_detection == 'AutoReject':
        print('Automatic bad-channel detection using AutoReject')
        bad_channel_detection = 'AutoReject'
    elif bad_channel_detection == 'RANSAC':
        print('Automatic bad-channel detection using RANSAC')
        bad_channel_detection = 'RANSAC'

    if bad_channel_detection:
        print('Performing automatic bad channel detection')
        detected_bad_channels = get_bad_channels(globals.raw, bad_channel_detection)
        # print(detected_bad_channels)

        total_bad_channels = globals.raw.info['bads']
        for bad_channel in detected_bad_channels:
            if bad_channel not in total_bad_channels:
                total_bad_channels.append(bad_channel)

        raw.info['bads'] = total_bad_channels

    # Re-referencing
    if reference:
        # print('Reference: {}'.format(reference))
        if reference == 'None':
            print('No re-referencing')
            reference = None
        elif reference != 'average':
            reference = [reference]

        if reference:
            print('Applying custom reference {}'.format(reference))
            raw.set_eeg_reference(reference)

    # Bad-channel interpolation
    if bad_channel_interpolation:
        # print(globals.raw.info['bads'])
        print('Performing bad-channel interpolation')
        raw = raw.interpolate_bads(reset_bads=False)
        
    return raw