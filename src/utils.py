import numpy as np
import matplotlib.pyplot as plt

def plot_iterations_2D(func, points, plot_x_lim=10, plot_y_lim=10, plot_nb_contours=10):
    xlist = np.linspace(-1*plot_x_lim, plot_x_lim, plot_nb_contours)
    ylist = np.linspace(-1*plot_y_lim, plot_y_lim, plot_nb_contours)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            Z[i, j] = func((X[i, j], Y[i, j]))
    plt.clf()
    cs = plt.contour(X, Y, Z, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    #Show contour values.
    #plt.clabel(cs, cs.levels, inline=True, fontsize=10)
    plt.scatter(points[0, 0], points[0, 1], marker='.', color='#495CD5')
    plt.scatter(points[1:, 0], points[1:, 1], marker='.', color='#5466DE')
    plt.scatter(points[-1, 0], points[-1, 1], marker='*', color='r')

def plot_iterations_1D(func, points, plot_x_lim=10, plot_nb_contours=10):
    X = np.linspace(-1*plot_x_lim, plot_x_lim, plot_nb_contours)
    Z = np.zeros_like(X)
    for i in range(0, len(X)):
        Z[i] = func(X[i])
    plt.clf()
    cs = plt.contour(X, Z, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    #Show contour values.
    #plt.clabel(cs, cs.levels, inline=True, fontsize=10)
    plt.scatter(points[0], marker='.', color='#495CD5')
    plt.scatter(points[1:], marker='.', color='#5466DE')
    plt.scatter(points[-1], marker='*', color='r')