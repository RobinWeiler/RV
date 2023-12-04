file_name = ''
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
disagreed_bad_channels = []

plotting_data = {}

preloaded_plots = {}
current_plot_index = 0

x0 = 0
x1 = 0
