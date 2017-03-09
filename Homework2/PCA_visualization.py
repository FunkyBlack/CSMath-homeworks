# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:29:19 2017

@author: FunkyBlack
"""
import numpy as np
import matplotlib.pyplot as plt

def visualize_raw_data(raw_data, digit_size = 32):
    
    figure = np.zeros((digit_size * 10, digit_size * 10))

    grid_x = 10
    grid_y = 10

    for i in range(grid_x):
        for j in range(grid_y):
            digit = raw_data[i + grid_x * j].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = (digit + 1) % 2
    
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='g')
    plt.style.use('grayscale')
    plt.grid(True, color='b', linestyle='-', linewidth=2)
    plt.imshow(figure)
    plt.title('The first 100 images of digit "3"')
    
    ax = plt.gca()
    ax.set_xticks(np.linspace(0,320,11))
    ax.set_xticklabels('')
    ax.set_yticks(np.linspace(0,320,11))
    ax.set_yticklabels('')
    plt.show()

def PCA_with_p_components(data, p):
    
    mean = np.mean(data, axis=0).reshape((1,-1))
    data_centered = data - mean
    U, S, V = np.linalg.svd(data_centered)
    W = np.dot(U[:,:p],np.diag(S[:p]))
    
    return W, V[:p], mean
    
def nearest_point(data, x, y):
    
    distance = np.sum(np.square(data - np.array([x,y])), axis=1)
    return np.argmin(distance)
    
def draw_data_with_PCA(data, V, mean, digit_size, raw_data):
    
    plt.scatter(data[:,0], data[:,1], color='g')
    figure = np.zeros((digit_size * 5, digit_size * 5))
    for p, i in enumerate(range(-4, 5, 2)):
        for k, j in enumerate(range(-4, 5, 2)):
            x = data[nearest_point(data, i, j), 0]
            y = data[nearest_point(data, i, j), 1]
            plt.scatter(x, y, edgecolors='r', cmap='w', linewidths=2)
#            construct_data = (np.dot(V.T, np.array([x, y]).reshape(2,1)) + mean.T).reshape(digit_size, digit_size)
            origin_data = raw_data[nearest_point(data, i, j)].reshape(digit_size, digit_size)
            figure[p * digit_size: (p + 1) * digit_size,
                   k * digit_size: (k + 1) * digit_size] = (origin_data + 1) % 2
    plt.grid()
    plt.title('PCA over digit "3" with 2 components')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
    plt.figure(figsize=(5, 5), facecolor='w', edgecolor='g')
    plt.style.use('grayscale')
    plt.imshow(figure)
    plt.grid(True, color='b', linestyle='-', linewidth=2)
    ax = plt.gca()
    ax.set_xticks(np.linspace(0,160,6))
    ax.set_xticklabels('')
    ax.set_yticks(np.linspace(0,160,6))
    ax.set_yticklabels('')
    plt.title('25 images corresponding to the red dots')
    plt.show()

def component_visual(mean, V, digit_size):
    fig, axs = plt.subplots(1, 3)
    
    ax = axs[0]
    figure = mean.reshape(digit_size, digit_size)
    ax.imshow(figure)
    
    ax = axs[1]
    figure = V[0].reshape(digit_size, digit_size)
    ax.imshow(figure)
    
    ax = axs[2]
    figure = V[1].reshape(digit_size, digit_size)
    ax.imshow(figure)

if __name__ == '__main__':
    p = 2
    digit_size = 32
    
    fname = 'all_digit_3.npy'
    raw_data = np.load(fname)
    print ("Numbers of digit '3': ", raw_data.shape[0])
    
    visualize_raw_data(raw_data[:100], digit_size=digit_size)
    W, V, mean = PCA_with_p_components(raw_data, p=p)
    draw_data_with_PCA(W, V, mean=mean, digit_size=digit_size, raw_data=raw_data)
    component_visual(mean, V, digit_size)