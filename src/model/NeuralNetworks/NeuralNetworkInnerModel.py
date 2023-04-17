import torch
import torch.nn as nn

class NeuralNetworkInnerModel(nn.Module):
  """
  A neural network to approximate the function h* for Neur. Imp. Diff.
  """
  def __init__(self, layer_sizes):
    super(NeuralNetworkInnerModel, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

  def forward(self, x):
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    return x