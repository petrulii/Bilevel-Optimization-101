import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


class FunctionApproximator():
	"""
	An object to approximate an arbitrary function.
	"""

	def __init__(self, layer_sizes=[2, 10, 20, 10, 1], batch_size = 64):
		"""
		Init method.
			param layer_sizes: sizes of the layers of the network used to approximate functions
			param batch_size: size of training batches
		"""
		self.batch_size = batch_size
		self.NN = NeuralNetwork(layer_sizes)

	def load_data(self, X_inner, y_inner, X_outer=None, y_outer=None):
		"""
		Loads numpy data into a type suitable for batch training.
			param X_inner: feature vectors of the data of the inner objective
			param y_inner: labels of the data of the inner objective
			param X_outer: feature vectors of the data of the outer objective
			param y_outer: labels of the data of the outer objective
		"""
		self.X_inner = X_inner
		self.y_inner = y_inner
		self.inner_data = Data(X_inner, y_inner)
		self.inner_dataloader = DataLoader(dataset=self.inner_data, batch_size=self.batch_size, shuffle=True)
		self.X_outer = X_outer
		self.y_outer = y_outer
		if not (self.X_outer is None and self.y_outer is None):
			self.outer_data = Data(X_outer, y_outer)
			self.outer_dataloader = DataLoader(dataset=self.outer_data, batch_size=self.batch_size, shuffle=True)

	def train(self, x_k = None, h_k = None, num_epochs = 100, learning_rate = 0.01):
		"""
		Trains a neural network that approximates a function.
			param x_k: current x_k used when approximating h*(x)
			param h_k: current h*(x) used when approximating a*(x)
			param num_epochs: number of training epochs
			param learning_rate: learning rate
		"""
		optimizer = torch.optim.SGD(self.NN.parameters(), lr=learning_rate)
		loss_values = []
		# If we are approximating h_*(x)
		if not (x_k is None and self.X_inner is None and self.y_inner is None) and (h_k is None and self.X_outer is None and self.y_outer is None):
			# Set the inner loss G with a fixed x as the objective function
			loss_fn = self.loss_G()
			for epoch in range(num_epochs):
				for X_i, y_i in self.inner_dataloader:
					# Zero all the parameter gradients
					#optimizer.zero_grad()
					pred = self.NN(X_i)
					loss = loss_fn(x_k, pred, y_i.unsqueeze(-1))
					loss_values.append(loss)
					loss.backward()
					optimizer.step()
		# If we are approximating a_*(x)
		elif not (h_k is None and self.X_inner is None and self.y_inner is None and self.X_outer is None and self.y_outer is None) and (x_k is None):
			# Set the loss H with a fixed h*(x) as the objective function
			loss_fn = self.loss_H()
			for epoch in range(num_epochs):
				for X_i, y_i, X_o, y_o in self.inner_dataloader, self.outer_dataloader: # FIX THE DATALOADER
					# Zero all the parameter gradients
					#optimizer.zero_grad()
					pred_i = self.NN(X_i)
					pred_o = self.NN(X_o)
					loss = loss_fn(h_k, pred_i, pred_o, y_i.unsqueeze(-1), y_o.unsqueeze(-1))
					loss_values.append(loss)
					loss.backward()
					optimizer.step()
		else:
			raise AttributeError("Can only approximate h*(x) or a*(x), you must provide necessary inputs")
		step = np.arange(0, len(loss_values), 1)
		fig, ax = plt.subplots(figsize=(8,5))
		plt.plot(step, np.array(loss_values))
		plt.title("Step-wise Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.show()

	# TODO
	def loss_G(self, x_k, pred_i, y_i):
		"""
		Returns a loss function to recover h*(x) that only depends on the output and the target.
		"""
		#sample_i = np.random.choice(self.X_inner, size=10)
		#sample_o = np.random.choice(self.X_outer, size=10)
		return torch.mean((pred_i - y_i)**2 + torch.mul(x_k,torch.sum(torch.pow(self.pred_i,2))))

	# TODO
	def loss_H(self, h_k, pred_i, pred_o, y_i, y_o):
		"""
		Returns a loss function to recover a*(x) that only depends on the output and the target.
		"""
		#sample_i = np.random.choice(self.X_inner, size=10)
		#sample_o = np.random.choice(self.X_outer, size=10)
		return 0

	def approximate_function(self): 
		"""
		Returns a function that approximates the true function.
		"""
		def f(x,y):
			value = torch.from_numpy(np.array([[x,y]]))
			return self.NN(value.float())
		return f


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