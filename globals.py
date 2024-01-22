username = None
external_save_file_path = None
parameters = {}

raw = None
viewing_raw = None
model_raw = None
external_raw = False

marked_annotations = []
model_annotation_label = 'bad_artifact_model'
annotation_label_colors = {'bad_artifact': 'red', model_annotation_label: 'red'}

bad_channels = {'current session': []}

plotting_data = {'EEG': {}, 'model': [], 'plot': {'x0': 0, 'x1': 0, 'disagreed_bad_channels': []}}  # , 'current_plot_index': 0}}
