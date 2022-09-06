![alt text](https://github.com/RobinWeiler/RV/blob/55ca35355978ad2bdbc9223a43fd470ecfc05ba2/assets/RV_logo.png)

# Welcome to Robin's Viewer (RV)

Robin's Viewer (RV) is an interactive web-application including a user-friendly EEG viewer with the main purpose of annotating time-series EEG data using deep-learning models as a decision-support-system.

## Documentation

To view RV's documentation, click [here](https://robinweiler.github.io/RV/) or open the "index.html" file in the "docs" directory using the webbrowser of your choice.

## Getting started

To use RV, clone this repository. Then, install the libraries listed in the "requirements.txt" file. RV was developed using Python 3.9.12.

You can either run RV directly or call it from within your own Python script.

### Performance

We highly recommend downsampling data before plotting, as this will have a big impact on the overall performance and usability of RV. This can be done using the parameter in the "Resampling rate" field of the "Preprocessing" window (e.g. to three times the lowpass-filter parameter, which is the default). Note that resampling will only be used for plotting and does not affect the sampling frequency of the data saved in the end. 

To further speed up the performance, we recommend segmenting the recording (e.g., into 60-second segments, which is the default). This can be changed through the "Segments (in seconds)" field of the same window. If this field is left empty, the entire signal will be plotted.

### Running RV directly

If you want to run RV as a stand-alone application, simply run the "RV.py" file, or the "run_RV_NB.ipynb" Jupyter Notebook, in the RV directory, and make sure your EEG files are located in the "data" directory.

### Running RV from a script

Place the "RV" directory into your project. Then, use the following lines of code after loading your EEG data in a MNE Raw object:

import sys
sys.path.insert(0, './RV')

from RV import run_viewer

run_viewer(your_raw_object)

## Demo

The "RV_demo.ipynb" Jupyter Notebook contains a demo of how RV can be used with external preprocessing. Some example data is provided at https://www.dropbox.com/sh/6llqont8px1s86b/AADmEONSZqmhFXfl5e7NcB8Ga?dl=0 . The "RV_demo_signal_preprocessed.fif" file was preprocessed using the same steps as outlined in the "RV_demo.ipynb" Jupyter Notebook (1-45Hz bandpass filter; 50Hz notch filter) and already has several marked bad channels.
