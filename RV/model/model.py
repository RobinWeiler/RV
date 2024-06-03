import os
import multiprocessing
from joblib import Parallel, delayed
from gpuparallel import GPUParallel
from gpuparallel import delayed as GPUdelayed

import numpy as np

import mne

import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool3d, Module
import torch.nn.functional as F

from RV.model.utils.smoothing import smooth_predictions


# Configuration for the CNN
MODEL_CONFIG = {
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


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        structure = MODEL_CONFIG['structure']
        input_image = MODEL_CONFIG['input']
        conv_params = MODEL_CONFIG['convolution_params']

        def convolution_output(h, w, kernel, stride, padding, pooling):
            output_size = (
            int((h + 2 * padding - kernel[0]) / stride + 1), int((w + 2 * padding - kernel[1]) / stride + 1))
            output_size = (int(output_size[0] / pooling[1]), int(output_size[1] / pooling[2]))

            return output_size

        # First layer
        self.cnn_layer1 = torch.nn.Sequential()
        self.cnn_layer1.add_module('Conv1',
                                   Conv2d(input_image['n_channels'],
                                          structure['input_groups'] * structure['filters1'],
                                          groups=structure['input_groups'], kernel_size=conv_params['kernel1'],
                                          stride=conv_params['stride1'], padding=conv_params['padding']))
        self.cnn_layer1.add_module('ReLU1', ReLU(inplace=True))
        self.cnn_layer1.add_module("MaxPool1", MaxPool3d(kernel_size=conv_params['pooling1'],
                                                         stride=conv_params['pooling1']))

        # Second layer
        self.cnn_layer2 = torch.nn.Sequential()
        self.cnn_layer2.add_module('Conv2',
                                   Conv2d(structure['input_groups'] * structure['filters1'],
                                          structure['input_groups'] * structure['filters2'],
                                          groups=structure['filters1'], kernel_size=conv_params['kernel2'],
                                          stride=conv_params['stride2'], padding=conv_params['padding']))
        self.cnn_layer2.add_module('ReLU2', ReLU(inplace=True))
        self.cnn_layer2.add_module("MaxPool2", MaxPool3d(kernel_size=conv_params['pooling2'],
                                                         stride=conv_params['pooling2']))

        # Third layer
        self.cnn_layer3 = torch.nn.Sequential()
        self.cnn_layer3.add_module('Conv3',
                                   Conv2d(structure['input_groups'] * structure['filters2'], structure['filters3'],
                                          kernel_size=conv_params['kernel3'], stride=conv_params['stride3'],
                                          padding=conv_params['padding']))
        self.cnn_layer3.add_module('ReLU3', ReLU(inplace=True))
        self.cnn_layer3.add_module("MaxPool3", MaxPool3d(kernel_size=conv_params['pooling3'],
                                                         stride=conv_params['pooling3']))

        # Calculate output size after the 1st convolution + ReLU + max pooling
        output1 = convolution_output(input_image['height'], input_image['width'],
                                     conv_params['kernel1'], conv_params['stride1'],
                                     conv_params['padding'], conv_params['pooling1'])
        # Calculate output size after the 2nd convolution + ReLU + max pooling
        output2 = convolution_output(output1[0], output1[1], conv_params['kernel2'],
                                     conv_params['stride2'], conv_params['padding'],
                                     conv_params['pooling2'])
        # Calculate output size after the 3rd convolution + ReLU + max pooling
        self.output3 = convolution_output(output2[0], output2[1], conv_params['kernel3'],
                                          conv_params['stride3'], conv_params['padding'],
                                          conv_params['pooling3'])

        # Define one fully connected linear layer
        self.linear_layers = Linear(structure['filters3'] * self.output3[0] * self.output3[1],
                                    structure['n_classes'])

        self.n_filters3 = structure['filters3']

    # Define the forward pass
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = x.view(-1, self.n_filters3 * self.output3[0] * self.output3[1])
        x = self.linear_layers(x)

        return x


def load_model(device: torch.device):
    """Loads the trained CNN model.

    Args:
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        net (ConvNet): Trained CNN with loaded weights.
    """
    model = ConvNet().to(device)

    # Load model weights
    current_path = os.path.dirname(os.path.abspath(__file__))
    path_to_model_weights = os.path.join(current_path, 'Diachenko_trained_model.pth.tar')
    model_weights = torch.load(path_to_model_weights, map_location=device)
    model.load_state_dict(model_weights['state_dict'])

    return model.double()

def feed_data_to_model(model: ConvNet, tf_tensor: torch.Tensor, device: torch.device):
    """Runs the CNN model on tf_tensor and returns predicted artifact probabilities.

    Args:
        model (Network.ConvNet): Weights of the CNN model trained on time-frequency-transformed EEG segments.
        tf_tensor (torch.Tensor): Time-frequency-transformed EEG segments of normalized power converted to torch.Tensor.
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        probabilities (numpy.array): Predicted class probabilities.
    """
    # Reshape TF tensor (n_segments, n_channels, n_freq, n_times) to (n_segments, 1, n_channels, n_freq, n_times)
    tf_tensor = torch.reshape(tf_tensor, (tf_tensor.shape[0], 1, tf_tensor.shape[1],
                                          tf_tensor.shape[2], tf_tensor.shape[3]))

    print('Applying tfCNN ...')
    
    if device.type == 'cpu':
        num_cores = multiprocessing.cpu_count()
    elif device.type == 'cuda':
        num_cores = torch.cuda.device_count()

    def run(input):
        model.eval()
        with torch.no_grad():
            tf = input
            tf = tf.to(device)

            outputs = model(tf)

            prob = F.softmax(outputs, dim=1)

        return prob

    if device.type == 'cpu':
        probabilities = Parallel(n_jobs=num_cores, backend='threading')(delayed(run)(tf_tensor[k]) for k in range(len(tf_tensor)))
    elif device.type == 'cuda':
        probabilities = GPUParallel(n_gpu=num_cores)(GPUdelayed(run)(tf_tensor[k]) for k in range(len(tf_tensor)))

    probabilities = torch.squeeze(torch.stack(probabilities))

    return probabilities.cpu().numpy()

def model_predict(TF_data: torch.Tensor, segments: mne.Epochs, model: ConvNet, device: torch.device):
    """Feeds data to the CNN model.

    Args:
        TF_data (torch.Tensor): EEG segments transformed to TF images.
        segments (mne.Epochs): EEG segments as mne.Epochs.
        model (Network.ConvNet): Trained CNN with loaded weights.
        device (torch.device): Device on which to compute on. Can be torch.device('cpu') or torch.device('cuda').

    Returns:
        predictions (np.array): CNN predictions as artifact probability.
    """
    predictions = feed_data_to_model(model, TF_data, device)

    segments._metadata.reset_index(inplace=True)
    times = [np.arange(segments._metadata['start'].tolist()[i],
                       segments._metadata['end'].tolist()[i] + 1 / segments._metadata['sfreq'].tolist()[i],
                       1 / segments._metadata['sfreq'].tolist()[i]) for i in segments._metadata.index.tolist()]

    print('Interpolating and smoothing predictions ...')
    predictions = smooth_predictions(predictions, times)
    print(f'Amount of prediction points: {len(predictions)}')

    return predictions
