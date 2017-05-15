# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:51:36 2017

@author: FunkyBlack
"""
import numpy as np
import matplotlib.pyplot as plt

def obj_fun(A, b, x):
#==============================================================================
#   define the test function
#                       T      T
#             T       -b x   -x x
#     f(x) = x A x + e    + e
# 
#==============================================================================
    f_x = np.dot(np.dot(x.T, A), x) + np.exp(np.dot(-b.T, x)) + np.exp(np.dot(-x.T, x))
    return f_x

def grad_fun(A, b, x):
    # Compute the gradient
    g_x = 2 * np.dot(A, x) - np.exp(np.dot(-b.T, x)) * b - 2 * np.exp(np.dot(-x.T, x)) * x
    return g_x
    
def hess_fun(A, b, x):
    # Compute the Hessian matrix
    G_x = 2 * A + np.exp(np.dot(-b.T, x)) * np.dot(b, b.T) - 2 * np.exp(np.dot(-x.T, x)) * np.eye(5) + 4 * np.exp(np.dot(-x.T, x)) * np.dot(x, x.T)
    return G_x
    
def LevMar(A, b):
    Max_iter = 100
    epsilon = 1e-8
    
    x = np.array([[0],[0],[0],[0],[0]])
    mu = 0.01
    
    x_plot = []
    y_plot = []

    for i in range(Max_iter):
        f_x = obj_fun(A=A, b=b, x=x)
        g_x = grad_fun(A=A, b=b, x=x)
        G_x = hess_fun(A=A, b=b, x=x)
        
        x_plot.append(i)
        y_plot.append(f_x[0])
        
        if np.sum(np.abs(g_x)) < epsilon:
            break
        EigVal = np.linalg.eigvals(G_x + mu * np.eye(5))
        while(np.all(EigVal > 0) == False):
            mu = 4 * mu
            EigVal = np.linalg.eigvals(G_x + mu * np.eye(5))
        s = np.dot(-np.linalg.inv(G_x + mu * np.eye(5)), g_x)
        f_x_new = obj_fun(A=A, b=b, x=x+s)
        delta_q = np.dot(g_x.T, s) + 0.5 * np.dot(np.dot(s.T, G_x), s)
        r = (f_x_new - f_x) / delta_q
        if r < 0.25:
            mu = 4 * mu
        elif r > 0.75:
            mu = mu / 2
        if r > 0:
            x = x + s
            
    plt.plot(x_plot, y_plot, 'ko', x_plot, y_plot, 'r')
    xlims = plt.xlim()
    ylims = plt.ylim()
    plt.xlim(xlims[0]-0.5, xlims[1]+0.5)
    plt.ylim(ylims[0]-0.1, ylims[1]+0.1)
    plt.title('Levenberg-Marquardt Method')
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.text(3.5, 1.8, '$f(x)=x^{T}Ax+e^{-b^{T}x}+e^{-x^{T}x}$', fontsize=15)
    
    return x, f_x[0]
            
if __name__ == '__main__':
    A = np.array([[5,-1,2,0,0],
                  [-1,4,1,-1,0],
                  [2,1,6,4,0],
                  [0,-1,4,7,0],
                  [0,0,0,0,0.5]])
    b = np.array([[2],
                  [1],
                  [3],
                  [5],
                  [10]])
    x_star, f_x_star = LevMar(A=A, b=b)
    print ("x* = ", x_star, "\nvalue = ", f_x_star)
    plt.savefig('LM_iterations.png')