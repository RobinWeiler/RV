import torch
from torch.utils.data import SubsetRandomSampler

# Configuration for the final CNN
cfg = {
    'model': {
        'network': 'CNN'
    },

    'input': {
        'formulation': 'TF power',
        'height': 45, 'width': 100, 'n_channels': 19
    },

    'structure': {
        'n_layers': 3, 'input_groups': 19, 'input_padding': 0,
        'filters1': 50, 'filters2': 100, 'filters3': 150,
        'n_classes': 2
    },

    'convolution_params': {
        'kernel1': (1, 5), 'stride1': 1, 'pooling1': (1, 2, 2),
        'kernel2': (5, 5), 'stride2': 1, 'pooling2': (1, 2, 2),
        'kernel3': (3, 3), 'stride3': 1, 'pooling3': (1, 1, 1),
        'padding': 0
    },

    'training_params': {
        'lr': 0.0001, 'momentum': 0.9,
        'optimizer': ('SGD', torch.optim.SGD),
        'batch_size': 64, 'epochs': 70, 'criterion': torch.nn.CrossEntropyLoss()
    }
}