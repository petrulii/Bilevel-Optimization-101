import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class Data(Dataset):
    """
    A class for converting numpy data to torch tensors.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class NeuralNetwork(nn.Module):
    """
    A neural network to approximate an arbitrary function.
    """
    def __init__(self, layer_sizes):
        if len(layer_sizes) != 5:
            raise ValueError("The network has five layers, you must give a list with five integer values")
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        nn.init.kaiming_uniform_(self.layer_1.weight)
        self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        nn.init.kaiming_uniform_(self.layer_2.weight)
        self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        nn.init.kaiming_uniform_(self.layer_3.weight)
        self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])
       
    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.tanh(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        x = torch.tanh(self.layer_4(x))
        return x

class FunctionApproximator():
    """
    An object to approximate an arbitrary function.
    """
    def __init__(self, layer_sizes=[2, 10, 20, 10, 1], batch_size = 64):
        self.batch_size = batch_size
        self.NN = NeuralNetwork(layer_sizes)

    def load_data(self, X_inner, y_inner, X_outer=None, y_outer=None):
        self.X_inner = X_inner
        self.y_inner = y_inner
        self.inner_data = Data(X_inner, y_inner)
        self.inner_dataloader = DataLoader(dataset=self.inner_data, batch_size=self.batch_size, shuffle=True)
        self.X_outer = X_outer
        self.y_outer = y_outer
        if not (self.X_outer is None and self.y_outer is None):
            self.outer_data = Data(X_outer, y_outer)
            self.outer_dataloader = DataLoader(dataset=self.outer_data, batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs = 100, learning_rate = 0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        # If we are approximating h_*(x)
        if not (self.X_outer is None and self.y_outer is None):
            # Set the inner loss G with a fixed x as the objective function
            loss_fn = self.loss_G
        # If we are approximating a_*(x)
        else:
            # Set the loss H with a fixed x as the objective function
            loss_fn = self.loss_H
        optimizer = torch.optim.SGD(self.NN.parameters(), lr=self.learning_rate)
        loss_values = []
        for epoch in range(self.num_epochs):
            for X, y in self.train_dataloader: # FIX THE DATALOADER
                # Zero all the parameter gradients
                optimizer.zero_grad()
                pred = self.NN(X)
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss_values.append(loss)
                loss.backward()
                optimizer.step()
        step = np.arange(0, len(loss_values), 1)
        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(step, np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    
    def loss_G(self, output, target): # FIX THIS
        loss = torch.mean((output - target)**2 + torch.sum(self.X_inner)**2)
        return loss
    
    def loss_H(self, output, target): # FIX THIS
        loss = torch.mean((output - target)**2 + torch.sum(self.X_inner)**2)
        return loss
    
    def approximate_function(self): 
        def f(x,y):
            value = torch.from_numpy(np.array([[x,y]]))
            return self.NN(value.float())
        return f