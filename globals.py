external_save_file_path = None
parameters = {}

raw = None
viewing_raw = None
model_raw = None
external_raw = False

plotting_data = {
    'EEG': {},
    'model': [],
    'plot': {'x0': 0, 'x1': 0, 'disagreed_bad_channels': []},
    'annotations': {'bad_channels': {'current session': []}, 'marked_annotations': [], 'default_model_annotation_label': 'bad_artifact_model', 'annotation_label_colors': {'bad_artifact': 'red'}}
}
