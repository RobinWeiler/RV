
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool3d, Module


class ConvNet(Module):
    def __init__(self, cfg):
        super(ConvNet, self).__init__()

        structure = cfg['structure']
        input_image = cfg['input']
        conv_params = cfg['convolution_params']

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

