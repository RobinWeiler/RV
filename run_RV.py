import os
import shutil
import webbrowser

from dash_extensions.enrich import DashProxy, ServersideOutputTransform, NoOutputTransform, FileSystemBackend

from RV.layout import get_RV_layout
from RV.callbacks.annotation_callbacks import register_annotation_callbacks
from RV.callbacks.bad_channel_callbacks import register_bad_channel_callbacks
from RV.callbacks.channel_selection_callbacks import register_channel_selection_callbacks
from RV.callbacks.loading_callbacks import register_loading_callbacks
from RV.callbacks.offset_callbacks import register_offset_callbacks
from RV.callbacks.power_spectrum_callbacks import register_power_spectrum_callbacks
from RV.callbacks.quit_callbacks import register_quit_callbacks
from RV.callbacks.saving_callbacks import register_saving_callbacks
from RV.callbacks.scale_callbacks import register_scale_callbacks
from RV.callbacks.segment_callbacks import register_segment_callbacks
from RV.callbacks.shortcut_callbacks import register_shortcut_callbacks
from RV.callbacks.stats_callbacks import register_stats_callbacks
from RV.callbacks.topomap_callbacks import register_topomap_callbacks
from RV.callbacks.visualization_callbacks import register_visualization_callbacks


def run_RV(load_file_path=None, save_file_path=None, serverside_cache=None,
           disable_file_selection=False, disable_manual_saving=False, disable_preprocessing=False, disable_model=True,
           auto_save=False, port=8060,
           external_raw=None):
    """Run Robin's Viewer. Will open RV in browser at 'http://localhost:8060' by default.

    Args:
        load_file_path (str, optional): Path to directory from which to load files. Defaults to None ('data' directory in RV).
        save_file_path (str, optional): Path to directory to which to save files. Defaults to None ("save_files" directory in RV).
        serverside_cache (str, optional): Path to directory to which to save Serverside cache. Defaults to None ("file_system_backend" directory in RV).
        disable_file_selection (bool, optional): Whether or not to disable file selection and hide respective UI elements. Defaults to False.
        disable_manual_saving (bool, optional): Whether or not to disable manual saving and hide respective UI elements. Defaults to False.
        disable_preprocessing (bool, optional): Whether or not to disable in-app preprocessing and hide respective UI elements. Defaults to False.
        disable_model (bool, optional): Whether or not to disable integration of model predictions and hide respective UI elements. Defaults to True.
        auto_save (bool, optional): Whether or not to activate automatic saving. If True, saves loaded file to save_file_path after every change. Defaults to False.
        port (int, optional): Port at which to run RV (http://localhost:port). Defaults to 8060.
        external_raw (mne.io.Raw, optional): mne.io.Raw object to load in RV. If provided, disable_file_selection and disable_preprocessing become True. Defaults to None.
    """
    if external_raw != None and save_file_path is None:
        raise Exception('When running RV with external data, save_file_path has to be provided.')

    current_path = os.path.dirname(os.path.abspath(__file__))
    RV_path = os.path.join(current_path, 'RV')

    if load_file_path is None:
        load_file_path = os.path.join(RV_path, 'data')
    elif not os.path.exists(load_file_path):
        raise Exception(f'Given load_file_path {load_file_path} does not exist.')
    print(f'Loading data from {load_file_path}')

    if save_file_path is None:
        save_file_path = os.path.join(RV_path, 'save_files')
    elif not os.path.exists(save_file_path) and external_raw is None:
        raise Exception(f'Given save_file_path {save_file_path} does not exist.')
    print(f'Saving data to {save_file_path}')

    if external_raw != None:
        disable_file_selection = True
        disable_preprocessing = True

        temp_save_file_path = save_file_path
        external_raw.save(temp_save_file_path, overwrite=True)
    else:
        temp_save_file_path = os.path.join(RV_path, 'temp_raw.fif')
    print(f'Saving temporary saves to {temp_save_file_path}')

    if serverside_cache is None:
        serverside_cache = os.path.join(RV_path, 'file_system_backend')
        if not os.path.exists(serverside_cache):
            os.mkdir(serverside_cache)
    elif not os.path.exists(serverside_cache):
        raise Exception(f'Given serverside_cache {serverside_cache} does not exist.')

    # Clear Serverside cache
    for filename in os.listdir(serverside_cache):
        if filename == 'info.txt':
            continue
        else:
            file_path = os.path.join(serverside_cache, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    app = DashProxy(__name__,
                    assets_folder=os.path.join(RV_path, 'assets'),
                    transforms=[ServersideOutputTransform([FileSystemBackend(serverside_cache)])]
    )

    app.title = 'RV'

    app.layout = get_RV_layout(load_file_path, save_file_path, serverside_cache, temp_save_file_path,
                               disable_file_selection, disable_manual_saving, disable_model, disable_preprocessing,
                               auto_save
    )

    # Callbacks
    register_annotation_callbacks()
    register_bad_channel_callbacks()
    register_channel_selection_callbacks()
    register_loading_callbacks(auto_save, external_data=False if external_raw is None else True)
    register_offset_callbacks()
    register_power_spectrum_callbacks()
    register_quit_callbacks()
    register_saving_callbacks()
    register_scale_callbacks()
    register_segment_callbacks()
    register_shortcut_callbacks()
    register_stats_callbacks()
    register_topomap_callbacks()
    register_visualization_callbacks()

    if not disable_model:
        from RV.callbacks.model_callbacks import register_model_callbacks
        register_model_callbacks()

    # webbrowser.open(f'http://localhost:{port}')

    app.run_server(debug=True, dev_tools_ui=True, jupyter_mode='tab', port=port)


if __name__ == '__main__':
    run_RV()
