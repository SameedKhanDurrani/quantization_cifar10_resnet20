import torch
import torch.quantization
from resnet.resnet import ResNet, ResidualBlock


def create_model():
    return ResNet(ResidualBlock, [2, 2, 2])


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


def get_quantized_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


model = create_model()
load_model(model, 'resnet.ckpt')
quantized_model_8_bits = get_quantized_model(model)

