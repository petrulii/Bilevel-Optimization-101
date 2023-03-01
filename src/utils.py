import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_2D_functions(f1, f2, plot_x_lim=10, plot_y_lim=10, plot_nb_contours=10):
    """
    A function to plot two continuos 2D functions side by side on the same domain.
    	param f1: first function
    	param f2: second function
    """
    # Create a part of the domain.
    xlist = np.linspace(-1*plot_x_lim, plot_x_lim, plot_nb_contours)
    ylist = np.linspace(-1*plot_y_lim, plot_y_lim, plot_nb_contours)
    X, Y = np.meshgrid(xlist, ylist)
    # Get mappings from both the true and the approximated functions.
    Z1 = np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            Z1[i, j] = f1(X[i, j], Y[i, j])
    Z2 = np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            Z2[i, j] = f2(X[i, j], Y[i, j])
    # Visualize the true function.
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    #fig.clf()
    ax1.contour(X, Y, Z1, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    ax1.set_title("True function")
    ax1.set_xlabel("Feature #0")
    ax1.set_ylabel("Feature #1")
    # Visualize the approximated function.
    ax2.contour(X, Y, Z2, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    ax2.set_title("Approximated function")
    ax2.set_xlabel("Feature #0")
    ax2.set_ylabel("Feature #1")
    fig.show()
    fig.savefig("figures/2D_function_approximation.png")

def plot_loss(loss_values):
    """
    Plot the loss value over iterations.
    	param loss_values: list of values to be plotted
    """
    loss_values = [tensor.item() for tensor in loss_values]
    step = np.arange(0, len(loss_values), 1)
    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(step, loss_values)
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def sample(X, n):
    """
    Take a uniform sample of size n from tensor X.
    	param X: data tensor
    	param n: sample size
    """
    probas = torch.full([n], 1/n)
    index = (probas.multinomial(num_samples=n, replacement=True)).to(dtype=torch.long)
    return X[index]