![alt text](https://github.com/RobinWeiler/RV/blob/77ffeb669c2fd5eb0caaec49128fee72f0150642/RV/assets/RV_title_image.png)

# Welcome to Robin's Viewer (RV)

Robin's Viewer (RV) is a user-friendly EEG viewer with the main purpose of annotating time-series EEG data. It is implemented as an interactive web-application and is hence platform independent. RV also facilitates the use of deep-learning models as a decision-support-system.

## Getting started

To use RV, clone this repository. Then, install the libraries listed in the "requirements.txt" file. RV was developed using Python 3.11, so this is what we recommend to use.

You can run RV directly by running ```python run_RV.py``` in your terminal. RV will then start in your web-browser under http://localhost:8060 .

Alternatively, you can call the ```run_RV()``` function from the "run_RV.py" file from within your own Python script. To do this, place the "RV" directory into your project folder. Then, use the following lines of code after preprocessing and loading your EEG data into an mne.io.Raw object:

```
import sys
sys.path.insert(0, './RV')

from RV.run_RV import run_RV

run_RV(external_raw=YOUR_RAW, save_file_path=YOUR_SAVE_FILE_PATH)
```

```run_RV()``` allows to pass the following optional arguments:
<ul>
  <li>load_file_path (str, optional): Path to directory from which to load files. Defaults to the "data" directory in RV.</li>
  <li>save_file_path (str, optional): Path to directory to which to save files. Defaults to the "save_files" directory in RV.</li>
  <li>serverside_cache (str, optional): Path to directory to which to save Serverside cache. Defaults to the "file_system_backend" directory in RV.</li>
  <li>disable_file_selection (bool, optional): Whether or not to disable file selection and hide respective UI elements. Defaults to False.</li>
  <li>disable_manual_saving (bool, optional): Whether or not to disable manual saving and hide respective UI elements. Defaults to False.</li>
  <li>disable_preprocessing (bool, optional): Whether or not to disable in-app preprocessing and hide respective UI elements. Defaults to False.</li>
  <li>disable_model (bool, optional): Whether or not to disable integration of model predictions and hide respective UI elements. Defaults to True.</li>
  <li>auto_save (bool, optional): Whether or not to activate automatic saving. If True, saves loaded file to save_file_path after every change. Defaults to False.</li>
  <li>port (int, optional): Port at which to run RV (http://localhost:port). Defaults to 8060.</li>
  <li>external_raw (mne.io.Raw, optional): mne.io.Raw object to load in RV. If provided, disable_file_selection and disable_preprocessing become True. Defaults to None.</li>
</ul>

A demo EEG recording, in a file called "RV_demo_signal.fif", can be found at https://surfdrive.surf.nl/files/index.php/s/F8LZv1qm9Gd4PPb . This file was preprocessed with a 1-45 Hz bandpass filter and a 50 Hz notch filter.

### Implemented deep-learning model

The integrated deep-learning model, which can be used to automatically mark artifacts in RV, is the model described in Diachenko, Marina, et al. "Improved manual annotation of EEG signals through convolutional neural network guidance." Eneuro 9.5 (2022). If you want to use the model, also install the libraries listed in the "requirements_model.txt". Then, download the model's weights from https://surfdrive.surf.nl/files/index.php/s/F8LZv1qm9Gd4PPb . Finally, set the ```disable_model``` argument of ```run_RV()``` to True.

Importantly, the model applies the following preprocessing steps, independent of the preprocessing set by the user in RV, to generate its predictions: The signal is band-pass filtered in the range of 0.5 – 45 Hz using a Hamming window and a transition bandwidth of 0.5 Hz at the low cut-off frequency and 11.25 Hz at the high cut-off frequency. Bad channels are interpolated using spherical spline interpolation. Signal is then re-referenced to the average electrode, and the following 19 channels are selected from the standard 10-20 system: Fp1, F7, T3, T5, F3, C3, P3, O1, Fp2, F8, T4, T6, F4, C4, P4, O2, Fz, Cz, Pz. Then, the signal is segmented into 1-second segments with 50% overlap between consecutive windows, and each 19-channel EEG segment is transformed into a time-frequency plot using complex wavelet convolution. Morlet wavelets are constructed over 45 logarithmically-spaced frequency bins in the range of 0.5 – 45 Hz. The time resolution is specified as a function of frequency using a logarithmically spaced vector between 1.2 and 0.2 seconds. Wavelet convolution is performed per EEG channel, and the convolution output is resampled to 100 Hz along the time axis. Values are then normalized using Z-score normalization across all channels.

#### Using CUDA

By default, the model is run on the CPU to avoid potential CUDA driver issues. If your PyTorch installation works with the CUDA drivers of your GPU, you can un-comment line 10 in RV/model/run_model.py . This can speed up the model significantly.

## Performance

In order to improve performance, we implemented the plotly-resampler package (Van Der Donckt, Jonas, et al. "Plotly-resampler: Effective visual analytics for large time series." 2022 IEEE Visualization and Visual Analytics (VIS). IEEE, 2022.) in version 2.0 of RV. This allows for dynamic resampling of the visualized data at different levels of zoom. Specifically, the amount of datapoints plotted per trace in one given window is constant (set by default to the amount of pixels in width RV is using in the browser but customizable in the GUI) and hence the data is resampled every time the plotted window changes. When plotting the EEG signal in 10-second windows, 1,000 points plotted (per trace) hence results in a sampling rate of 100 Hz.

## Code documentation

To view RV's documentation, visit https://robinweiler.github.io/RV .

## Citation

If you use this software, please cite the paper listed below.

```bibtex
@article{weiler2023robin,
  title={Robin’s viewer: using deep-learning predictions to assist EEG annotation},
  author={Weiler, Robin and Diachenko, Marina and Juarez-Martinez, Erika L and Avramiea, Arthur-Ervin and Bloem, Peter and Linkenkaer-Hansen, Klaus},
  journal={Frontiers in Neuroinformatics},
  volume={16},
  pages={1025847},
  year={2023},
  publisher={Frontiers Media SA}
}
```

### Model paper

```bibtex
@article{diachenko2022improved,
  title={Improved manual annotation of EEG signals through convolutional neural network guidance},
  author={Diachenko, Marina and Houtman, Simon J and Juarez-Martinez, Erika L and Ramautar, Jennifer R and Weiler, Robin and Mansvelder, Huibert D and Bruining, Hilgo and Bloem, Peter and Linkenkaer-Hansen, Klaus},
  journal={Eneuro},
  volume={9},
  number={5},
  year={2022},
  publisher={Society for Neuroscience}
}
```