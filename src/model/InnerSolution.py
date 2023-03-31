import sys
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src/')

from model.utils import set_seed
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import plot_1D_iterations, plot_2D_functions, plot_loss


class InnerSolution(nn.Module):
  """
  Instanciates the inner solution of the bilevel problem.
  """

  def __init__(self, inner_objective, layer_sizes, device, batch_size=64):
    """
    Init method.
      param inner_objective: inner level objective function
      param layer_sizes: layer sizes of the network approximating h*
      param device: cpu or cuda
      param batch_size: batch size for training
    """
    self.__input_check__(inner_objective, layer_sizes, device, batch_size)
    super(InnerSolution, self).__init__()
    # Specify inner objective
    self.inner_objective = inner_objective
    # Specify network layers
    # Make this a network init function
    """self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])"""
    # Specify device to use during training
    self.device = device
    # Instanciate the neural network to approximate a*
    self.a = NeuralNetwork_a(layer_sizes, device).to(device)

  def __input_check__(self, inner_objective, layer_sizes, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (inner_objective is None):
      raise AttributeError("You must specify the inner objective.")
    if layer_sizes is None:
        raise AttributeError("You must specify the layer sizes for the network.")
    if len(layer_sizes) != 5:
        raise ValueError("Networks have five layers, you must give a list with five integer values.")
    if not (type(batch_size) is int):
        raise TypeError("Batch size must be an integer.")

  def forward(self, x, mu_k):
    """
    Forward pass of a neural network that approximates the function h* for Neur. Imp. Diff.
      param x: input to the neural network
    """
    # Train the model to approximate h* at mu_k
    self.train()
    # Give h*(x) value
    x = x.to(self.device)
    x = torch.relu(self.layer_1(x))
    x = torch.tanh(self.layer_2(x))
    x = torch.tanh(self.layer_3(x))
    x = self.layer_4(x)
    return x

  def backward(self, outer_objective):
    """
    Forward pass of a neural network that approximates the function h* for Neur. Imp. Diff.
      param x: input to the neural network
    """
    # Here for a() need to train on the same X_inner batches as h was trained on
    # and on X_outter that h was evaluated on in the outer loop where we optimize for
    # the outer objective g(mu_k, h_k).
    self.a.train()
    return x


class NeuralNetwork_a(nn.Module):
  """
  A neural network to approximate the function a* for Neur. Imp. Diff.
  """

  def __init__(self, layer_sizes, device, batch_size=64):
    """
    Init method.
      param layer_sizes: layer sizes of the network approximating h*
      param device: cpu or cuda
      param batch_size: batch size for training
    """
    super(NeuralNetwork_a, self).__init__()
        # Specify network layers
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        # Specify device to use during training
    self.device = device

  def forward(self, x):
    """
    Forward pass of a neural network that approximates the function a* for Neur. Imp. Diff.
      param x: input to the neural network
    """
    x = x.to(self.device)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    return x

  def train(inner_dataloader, outer_dataloader, outer_loss, optimizer, max_epochs):
    """
    Training loop of a neural network that approximates the function a* for Neur. Imp. Diff.
      param dataloader: input to the neural network
    """
    for epoch in range(max_epochs):
      for i, ((X_i, y_i), (X_o, y_o)) in enumerate(zip(inner_dataloader, outer_dataloader)):
        # Compute prediction and loss
        pred = forward(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  def loss_H(self, mu_k, h_k, a, inner_grad22, outer_grad2, X_i, y_i, X_o, y_o):
      """
      Returns a loss function to recover a*(x) that only depends on the output and the target.
      """
      a_in = a(X_i)
      aT_in = a_in.T
      aT_out = a(X_o).T
      hess = inner_grad22(mu_k, h_k, X_i, y_i)
      grad = outer_grad2(mu_k, h_k, X_o, y_o)
      return (1/2)*torch.mean(aT_in @ hess @ a_in)+(1/2)*torch.mean(aT_out @ grad)