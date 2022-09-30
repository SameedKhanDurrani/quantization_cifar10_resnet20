import torch
import torch.quantization
import numpy as np

from resnet.resnet import ResNet, ResidualBlock


def create_model():
    return ResNet(ResidualBlock, [2, 2, 2])


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


def scale_tensor(tensor):
    tensor = np.array(tensor)
    return np.around((tensor - min(tensor)) * ((2**8)-1) / (max(tensor) - min(tensor)))


model = create_model()
load_model(model, 'resnet.ckpt')

print(scale_tensor(model.state_dict()['conv.weight'][0][0][0]))


