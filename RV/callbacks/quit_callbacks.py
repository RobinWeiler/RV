import os
import shutil

from dash_extensions.enrich import Output, Input, State, callback
from dash.exceptions import PreventUpdate


def register_quit_callbacks():
    @callback(
        Output('RV-clear-main-graph-button', 'n_clicks'),
        Input({'type': 'process-button', 'modal': 'RV-quit'}, 'n_clicks'),
        State('RV-file-paths', 'data'),
        prevent_initial_call=True
    )
    def quit_RV(quit, file_paths):
        """Empties Serverside cache and clicks RV-clear-main-graph-button (loads title image back into main-graph).
        """
        if quit:
            # Clear Serverside cache
            serverside_cache = file_paths['serverside_cache']
            if os.path.exists(serverside_cache):
                for filename in os.listdir(serverside_cache):
                    if filename == 'info.txt':
                        continue
                    else:
                        file_path = os.path.join(serverside_cache, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            return 1

        raise PreventUpdate
