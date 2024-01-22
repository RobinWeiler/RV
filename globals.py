external_save_file_path = None
parameters = {}

raw = None
viewing_raw = None
model_raw = None
external_raw = False

marked_annotations = []
annotation_label_colors = {'bad_artifact': 'red'}

bad_channels = {'current session': []}

plotting_data = {'EEG': {}, 'model': [], 'plot': {'x0': 0, 'x1': 0, 'disagreed_bad_channels': []}}  # , 'current_plot_index': 0}}
