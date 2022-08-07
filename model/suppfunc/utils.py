
import torch
from torch.autograd import Variable


# Transform to pad channel input dimension of the input image
class PaddingChannelsTransform:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        print(11, len(img[0]), len(img[0][0]))
        padding = Variable(torch.zeros(self.pad, len(img[0]), len(img[0][0])))
        return torch.cat((img, padding), 0)