from layout import setup_app
import globals

# app = setup_app()

def run_viewer(eeg_data=None, save_file_path=None, parameters_to_load=None):
    """Run Robin's Viewer with optional arguments. Will open RV in browser at "http://localhost:8050".

    Args:
        eeg_data (mne.io.Raw, optional): Pass data if preprocessed externally. Will disable file selection. Defaults to None.
        save_file_path (string, optional): Path to save annotated data to when using "Quit" button. Defaults to None.
        parameters_to_load (dict, optional): Dictionary with parameters to fill in upon loading. 
        Possible keys for parameters_to_load: 'username', 'high_pass', 'low_pass', 'reference', 'resampling_rate', 'scale', 'offset', 'segment_size'. Defaults to None.
    """
    disable_preprocessing_parameters = False
    disable_file_selection = False

    if eeg_data:
        print('Setting raw to {}'.format(eeg_data))
        print(eeg_data.info)

        globals.raw = eeg_data.copy()
        globals.external_raw = True
        globals.external_save_file_path = save_file_path

    if parameters_to_load:
        globals.parameters = parameters_to_load
    else:
        globals.parameters = {}

    app = setup_app(disable_file_selection, disable_preprocessing_parameters)

    app.run_server(debug=True, mode='external', port=8050)
    
    # webbrowser.open("http://localhost:{}".format(8050))

if __name__ == '__main__':
    run_viewer()
