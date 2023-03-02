import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_2D_functions(f1, f2, f3, points=None, plot_x_lim=[-5,5], plot_y_lim=[-5,5], plot_nb_contours=10):
    """
    A function to plot three continuos 2D functions side by side on the same domain.
    """
    # Create a part of the domain.
    xlist = np.linspace(plot_x_lim[0], plot_x_lim[1], plot_nb_contours)
    ylist = np.linspace(plot_y_lim[0], plot_y_lim[1], plot_nb_contours)
    X, Y = np.meshgrid(xlist, ylist)
    # Get mappings from both the true and the approximated functions.
    Z1, Z2, Z3 = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            a = np.array([X[i, j], Y[i, j]], dtype='float32')
            Z1[i, j] = f1(torch.from_numpy(a)).float()
            Z2[i, j] = f2(torch.from_numpy(a)).float()
            Z3[i, j] = f3(torch.from_numpy(a)).float()
    # Visualize the true function.
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15, 5))
    ax1.contour(X, Y, Z1, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    ax1.set_title("True function")
    ax1.set_xlabel("Feature #0")
    ax1.set_ylabel("Feature #1")
    # Visualize the approximated function.
    ax2.contour(X, Y, Z2, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    if not (points is None):
        ax2.scatter(points[:,0], points[:,1], marker='.')
    ax2.set_title("Classical Imp. Diff.")
    ax2.set_xlabel("Feature #0")
    ax2.set_ylabel("Feature #1")
    ax3.contour(X, Y, Z3, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    ax3.set_title("Neural Imp. Diff.")
    ax3.set_xlabel("Feature #0")
    ax3.set_ylabel("Feature #1")
    plt.show()


def plot_1D_iterations(iters1, iters2, f1, f2, plot_x_lim=[0,1]):
    """
    A function to plot three continuos 2D functions side by side on the same domain.
    """
    # Create a part of the domain.
    X = np.linspace(plot_x_lim[0], plot_x_lim[1], 100)
    # Visualize the true function.
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    Z1, Z2 = np.zeros_like(X), np.zeros_like(X)
    for i in range(0, len(X)):
        a = torch.from_numpy(np.array(X[i], dtype='float32'))
        Z1[i] = f1(a).float()
        Z2[i] = f2(a).float()
    Z3, Z4 = np.zeros(len(iters1)), np.zeros(len(iters2))
    X1, X2 = np.zeros(len(iters1)), np.zeros(len(iters2))
    for i in range(0, len(iters1)):
        a = iters1[i]
        X1[i] = a.float()
        Z3[i] = f1(a).float()
    for i in range(0, len(iters2)):
        a = iters2[i]
        X2[i] = a.float()
        Z4[i] = f2(a).float()
    ax1.plot(X, Z1, color='red')
    ax1.scatter(X1, Z3, marker='.')
    ax1.set_title("Classical Imp. Diff.")
    ax1.set_xlabel("\mu")
    ax1.set_ylabel("f(\mu)")
    ax2.plot(X, Z2, color='red')
    ax2.scatter(X2, Z4, marker='.')
    ax2.set_title("Neural Imp. Diff.")
    ax2.set_xlabel("\mu")
    ax2.set_ylabel("f(\mu)")
    plt.show()

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

def sample_X(X, n):
    """
    Take a uniform sample of size n from tensor X.
    	param X: data tensor
    	param n: sample size
    """
    probas = torch.full([n], 1/n)
    index = (probas.multinomial(num_samples=n, replacement=True)).to(dtype=torch.long)
    return X[index]

def sample_X_y(X, y, n):
    """
    Take a uniform sample of size n from tensor X.
    	param X: data tensor
    	param y: true value tensor
    	param n: sample size
    """
    probas = torch.full([n], 1/n)
    index = (probas.multinomial(num_samples=n, replacement=True)).to(dtype=torch.long)
    return X[index], y[index]