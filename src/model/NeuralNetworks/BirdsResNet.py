import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Use a pretrained model
        self.resnet34 = models.resnet34()
        # Get last layer input shape
        input_shape = self.resnet34.fc.in_features
        # Replace last layer with 200+312 output
        self.resnet34.fc = nn.Linear(input_shape, 512)
    
    def forward(self, x):
        x = self.resnet34(x)
        main_pred = x[:, 0:200]
        aux_pred = x[:, 200:512]
        return (main_pred, aux_pred)