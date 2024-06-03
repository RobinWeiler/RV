import math
import multiprocessing

from joblib import Parallel, delayed

import numpy as np

import scipy
import scipy.stats
from scipy.spatial import KDTree

import mne

from gpuparallel import GPUParallel
from gpuparallel import delayed as GPUdelayed

import torch


def wavelet_fft(frequencies: np.array, fwhm: np.array, nConv: int, wavtime: np.array):
    """Returns FFTs of complex Morlet wavelet.

    Args:
        frequencies (numpy.array): Frequency bins.
        fwhm (numpy.array): Full-width at half maximum parameter values which define time resolution.
        nConv (int): Length of the output. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros.
        wavtime (numpy.array): Wavelet time-series.

    Returns:
        waveletFFT (numpy.array): FFTs of complex Morlet wavelet.
    """
    waveletFFT = np.zeros([len(frequencies), nConv], dtype=np.complex_)

    for i, freq, width in zip(range(len(frequencies)), frequencies, fwhm):
        # Define gaussian
        gwin = np.exp((-4 * math.log(2) * (wavtime ** 2)) / (width ** 2))

        # Construct a complex Morlet wavelet
        cmw = np.exp(1j * 2 * np.pi * freq * wavtime) * gwin

        # FFT of the wavelet
        tempX = scipy.fft.fft(cmw, nConv)

        # Normalize
        norm = tempX[np.where(np.abs(tempX) == np.max(np.abs(tempX)))[0][0]]
        waveletFFT[i, :] = tempX / norm

    return waveletFFT

def wavelet_convolution(waveletFFT: np.array, segments: mne.Epochs, halfwav: int, nFreq: int, donwsampling: int, device: torch.device):
    """Returns Wavelet convolution results.

    Args:
        waveletFFT (np.array): FFT of a complex Morlet wavelet.
        segments (mne.Epochs): Segments of the EEG recording.
        halfwav (int): Index of the center point of the wavelet time-series.
        nFreq (int): Number of frequency bins.
        downsampling (int): Frequency (in Hz) to downsample to.
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        TFs (list): Z-score-normalized time-frequency power values flattened across channels for each EEG segment.
    """
    # Window size in seconds
    window_size = segments.times[-1] + (1 / segments.info['sfreq'])

    # Donwsampling step in seconds
    step = 1 / donwsampling

    # Create time bins for downsampling within the length of the window
    timeBins = np.arange(0.0, window_size, step)

    nChan = len(segments.ch_names)

    # Create indexing for times to look up the nearest neighbors of any point
    kdtree = KDTree(np.reshape(segments.times, (np.shape(segments.times)[0], 1)))

    # Find indices of the nearest points in kd-tree of times to time bin points
    _, inds = kdtree.query(np.reshape(timeBins, (np.shape(timeBins)[0], 1)))
    selectedTimeBins = inds

    if device.type == 'cpu':
        num_cores = multiprocessing.cpu_count()
    elif device.type == 'cuda':
        num_cores = torch.cuda.device_count()

    def processInput(input):
        data = input.get_data(copy=True)[0]
        tf = np.zeros([nFreq * nChan, len(timeBins)])

        rowIdx = 0
        for n in range(nChan):
            # FFT of the signal
            dataX1 = scipy.fft.fft(data[n], len(waveletFFT[0]))

            for j in range(nFreq):
                # print(n, j, waveletFFT[j, :])
                as1 = scipy.fft.ifft(waveletFFT[j, :] * dataX1)
                as1 = as1[halfwav:len(as1) - halfwav]
                tf[rowIdx, :] = (np.abs(as1[selectedTimeBins])) ** 2
                rowIdx = rowIdx + 1

        return scipy.stats.zscore(tf, axis=None)  # Z-score normalize

    if device.type == 'cpu':
        TFs = Parallel(n_jobs=num_cores, backend='threading')(delayed(processInput)(segments[k]) for k in range(len(segments)))
    elif device.type == 'cuda':
        TFs = GPUParallel(n_gpu=num_cores)(GPUdelayed(processInput)(segments[k]) for k in range(len(segments)))

    TFs = list(TFs)

    return TFs

def segment_TF(segments: mne.Epochs, device: torch.device):
    """Returns 8-bit unsigned integer values of intensity images of time-frequency-converted EEG segments.

    Args:
        segments (mne.Epochs): Segments of the EEG recording.
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        intensityTFs (list): 8-bit unsigned integer values of intensity images.
    """
    # Parameters for Wavelet convolution
    minFreq = 0.5  # Hz
    maxFreq = 45  # Hz
    nFreq = 45
    minFWHM = 0.2  # time resolution in seconds
    maxFWHM = 1.2  # time resolution in seconds

    # Get all frequencies
    frequencies = np.logspace(math.log10(minFreq), math.log10(maxFreq), nFreq)

    # Get time resolution for each frequency
    fwhm = np.logspace(math.log10(maxFWHM), math.log10(minFWHM), nFreq)

    # Wavelet parameters
    t = 4  # length in seconds
    # Wavelet centered at 0
    wavtime = np.linspace(-2, 2, t * int(segments.info['sfreq']) + 1)
    # Find the center of the wavelet (index)
    halfwav = list(np.where(wavtime == 0)[0])[0]

    # Convolution parameters
    nWave = len(wavtime)
    nData = len(segments.times)
    nConv = nWave + nData - 1

    # Complex Morlet wavelet FFTs
    cmwFFT = wavelet_fft(frequencies, fwhm, nConv, wavtime)

    # A list of time-frequency normalized power of EEG segments of the recording
    segment_TFs = wavelet_convolution(cmwFFT, segments, halfwav, len(frequencies), 100, device)

    # Convert to intensities
    intensityTFs = [(tf - np.min(tf)) / (np.max(tf) - np.min(tf)) for tf in segment_TFs]  # values between 0 and 1
    # Convert to uint8 image format
    intensityTFs = [(ints / np.max(ints))*255 for ints in intensityTFs]

    return intensityTFs
