import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader


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
		Loads data into a type suitable for batch training.
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

	def train(self, inner_grad22=None, outer_grad2=None, mu_k = None, h_k = None, num_epochs = 10, learning_rate = 0.001):
		"""
		Trains a neural network that approximates a function.
			param mu_k: current mu_k used when approximating h*(x)
			param h_k: current h*(x) used when approximating a*(x)
			param num_epochs: number of training epochs
			param learning_rate: learning rate
		"""
		optimizer = torch.optim.SGD(self.NN.parameters(), lr=learning_rate)
		loss_values = []
		# If we are approximating h_*(x)
		if not (mu_k is None and self.X_inner is None and self.y_inner is None) and (h_k is None and self.X_outer is None and self.y_outer is None):
			# Set the inner loss G with a fixed x as the objective function
			for epoch in range(num_epochs):
				for X_i, y_i in self.inner_dataloader:
					# Zero all the parameter gradients
					optimizer.zero_grad()
					h_k = self.NN
					loss = self.loss_G(mu_k, h_k, X_i, y_i)
					loss_values.append(loss.float())
					loss.backward()
					optimizer.step()
			return self.NN, loss_values
		# If we are approximating a_*(x)
		elif not (mu_k is None and h_k is None and self.X_inner is None and self.y_inner is None and self.X_outer is None and self.y_outer is None):
			# Set the loss H with a fixed h*(x) as the objective function
			for epoch in range(num_epochs):
				for i, ((X_i, y_i), (X_o, y_o)) in enumerate(zip(self.inner_dataloader, self.outer_dataloader)): # FIX THE DATALOADER
					# Zero all the parameter gradients
					optimizer.zero_grad()
					a_k = self.NN
					loss = self.loss_H(mu_k, h_k, a_k, inner_grad22, outer_grad2, X_i, y_i, X_o, y_o)
					loss_values.append(loss.float())
					loss.backward()
					optimizer.step()
			return self.NN, loss_values
		else:
			raise AttributeError("Can only approximate h*(x) or a*(x), you must provide necessary inputs")

	def loss_G(self, mu_k, h_k, X_i, y_i):
		"""
		Returns a loss function to recover h*(x) that only depends on the output and the target.
		"""
		# Here pred_h is a set of predictions h(x) for a set of w in an idd sample from p(w)
		return torch.mean(torch.pow((h_k(X_i) - y_i),2) + torch.mul(mu_k,torch.sum(torch.pow(h_k(X_i),2))))

	def loss_H(self, mu_k, h_k, a_k, inner_grad22, outer_grad2, X_i, y_i, X_o, y_o):
		"""
		Returns a loss function to recover a*(x) that only depends on the output and the target.
		"""
		# Here pred_a is a set of predictions a(w) for a set of w in an idd sample from p(w)
		aT_in = torch.transpose(a_k(X_i),0,1)
		hessian = inner_grad22(mu_k, h_k, X_i, y_i)
		aT_hessian = torch.matmul(aT_in.double(), hessian.double())
		a_in = a_k(X_i)
		aT_out = torch.transpose(a_k(X_o),0,1)
		grad = outer_grad2(mu_k, h_k, X_o, y_o)
		return (1/2)*torch.mean(torch.matmul(aT_hessian.double(), a_in.double()))+(1/2)*torch.mean(torch.matmul(aT_out.double(),grad.double()))


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
		x = self.layer_4(x)
		return x


class Data(Dataset):
	"""
	A class for data.
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.len = len(self.y)

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len