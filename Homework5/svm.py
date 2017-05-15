# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:49:10 2017

@author: FunkyBlack
"""
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt

def load_data(fname):
    with open(fname, 'r') as f:
        X = []
        y = []
        for line in f.readlines():
            data = line.split('\t')
            X.append(data[:2])
            y.append(data[-1])
        X = np.asarray(X).astype('float32')
        y = np.asarray(y).astype('int32')
    return X, y

def data_split(X, y):
    n_samples = X.shape[0]
    split = int(0.8 * n_samples)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test


def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1):
    return np.exp(-np.linalg.norm(x - y) ** 2 / 2 * (sigma ** 2))

def polynomial_kernel(x, y, d=2):
    return (np.dot(x, y) + 1) ** d

def kernel_matrix(X, kernel='linear', sigma=1, d=2):
    n_samples, n_features = X.shape
    K_matrix = np.zeros((n_samples, n_samples), dtype=float)
    if kernel == 'linear':
        for i in range(n_samples):
            for j in range(n_samples):
                K_matrix[i, j] = linear_kernel(X[i], X[j])
    elif kernel == 'rbf':
        for i in range(n_samples):
            for j in range(n_samples):
                K_matrix[i, j] = rbf_kernel(X[i], X[j], sigma=sigma)
    elif kernel == 'polynomial':
        for i in range(n_samples):
            for j in range(n_samples):
                K_matrix[i, j] = polynomial_kernel(X[i], X[j], d=d)
    else:
        print ('Kernel not in (linear, rbf, polynomial)')
    
    return K_matrix

def train_svm(X, y, C=1, kernel='linear', sigma=1, d=2):
    epsilon = 1e-5
    n_samples, n_features = X.shape
    K_matrix = kernel_matrix(X, kernel=kernel)
    P = matrix(np.outer(y, y) * K_matrix)
    q = matrix(-1 * np.ones(n_samples))
    G = matrix(np.concatenate((np.diag(np.ones(n_samples) * -1), np.diag(np.ones(n_samples)))))
    h = matrix(np.concatenate((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = matrix(y.astype('double'), (1, n_samples))
    b = matrix(0.0)
    # Using cvxopt to solve the qp
    sol = qp(P, q, G, h, A, b)
    alpha = np.ravel(sol['x'])
    
    sv = alpha > epsilon
    alpha_sv = alpha[sv]
    X_sv = X[sv]
    y_sv = y[sv]
    print ('Number of support vectors in data: ', len(alpha_sv))
    
    # Calculate b* using an alpha satisfies 0 < alpha < C
    for i in range(len(alpha_sv)):
        if alpha_sv[i] < (C - epsilon):
            temp_i = i
            break
    b_star = y_sv[temp_i]
    for j in range(len(alpha_sv)):
        if kernel == 'linear':
            b_star -= alpha_sv[j] * y_sv[j] * linear_kernel(X_sv[j], X_sv[temp_i])
        if kernel == 'rbf':
            b_star -= alpha_sv[j] * y_sv[j] * rbf_kernel(X_sv[j], X_sv[temp_i], sigma=sigma)
        if kernel == 'polynomial':
            b_star -= alpha_sv[j] * y_sv[j] * polynomial_kernel(X_sv[j], X_sv[temp_i], d=d)
    
    return alpha_sv, b_star, X_sv, y_sv

def test_svm(X, y, alpha_sv, b_star, X_sv, y_sv, C=1, kernel='linear', sigma=1, d=2):
    n_samples, n_features = X.shape
    y_pred = np.zeros_like(y)
    for i in range(n_samples):
        affine_func = b_star
        for j in range(len(X_sv)):
            if kernel == 'linear':
                affine_func += alpha_sv[j] * y_sv[j] * linear_kernel(X_sv[j], X[i])
            if kernel == 'rbf':
                affine_func += alpha_sv[j] * y_sv[j] * rbf_kernel(X_sv[j], X[i], sigma=sigma)
            if kernel == 'polynomial':
                affine_func += alpha_sv[j] * y_sv[j] * polynomial_kernel(X_sv[j], X[i], d=d)
        y_pred[i] = np.sign(affine_func)
    accuracy = np.sum(y_pred == y) / n_samples
    print ("Test accuracy %.4f%%: " % (accuracy * 100))
    
    # Plot the hyperplane
    plt.figure(figsize=(10,10))
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    x_grid = np.c_[xx.ravel(), yy.ravel()]

    y_grid = np.ones(x_grid.shape[0])

    for i in range(x_grid.shape[0]):
        affine_func = b_star
        for j in range(len(X_sv)):
            if kernel == 'linear':
                affine_func += alpha_sv[j] * y_sv[j] * linear_kernel(X_sv[j], x_grid[i])
            if kernel == 'rbf':
                affine_func += alpha_sv[j] * y_sv[j] * rbf_kernel(X_sv[j], x_grid[i], sigma=sigma)
            if kernel == 'polynomial':
                affine_func += alpha_sv[j] * y_sv[j] * polynomial_kernel(X_sv[j], x_grid[i], d=d)
        y_grid[i] = np.sign(affine_func)

    y_grid = y_grid.reshape(xx.shape)
    plt.pcolormesh(xx, yy, y_grid, cmap=plt.cm.Paired)
    
    # Plot test results
    for i, label in enumerate(y): 
        color[i] = 'brown' if label > 0 else 'blue'
    plt.scatter(X[:,0], X[:,1], marker='x', c=list(color.values()))
    return accuracy
    
def plot_data(X, y, X_sv, y_sv, accuracy, kernel = 'linear'):
    # Plot training data and support vectors in them
    color = {}
    for i, label in enumerate(y): 
        color[i] = 'brown' if label > 0 else 'blue'
    plt.scatter(X[:,0], X[:,1], marker='o', c=list(color.values()))
    for i, label in enumerate(y_sv):
        color[i] = 'brown' if label > 0 else 'blue'
    plt.scatter(X_sv[:,0], X_sv[:,1], marker='o', edgecolors='black', facecolor='none', s=55)
    plt.title('SVM using {} kernel (accuracy = {}% )'.format(kernel, str(accuracy * 100)))
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.show()
    
if __name__ == '__main__':
    fname = ('dataset.txt')
    C = 50
    kernel = 'polynomial'
    sigma = 1
    d = 2
    
    X, y = load_data(fname)
    X_train, y_train, X_test, y_test = data_split(X, y)
    alpha_sv, b_star, X_sv, y_sv = train_svm(X_train, y_train, C=C, kernel=kernel, sigma=sigma, d=d)
    accuracy = test_svm(X=X_test, y=y_test, C=C, kernel=kernel, sigma=sigma, d=d, alpha_sv=alpha_sv, b_star=b_star, X_sv=X_sv, y_sv=y_sv)
    plot_data(X=X_train, y=y_train, X_sv=X_sv, y_sv=y_sv, accuracy=accuracy, kernel=kernel)
    plt.savefig('SVM_visual_{}.png'.format(kernel))
    