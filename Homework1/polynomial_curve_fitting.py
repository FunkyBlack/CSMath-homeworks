# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:38:19 2017

@author: Ming Chen
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def func(x):
    return np.sin(2 * np.pi * x)
    
def gen_data(p_num):
    x = np.linspace(0, 1, p_num)
    y = func(x) + np.random.normal(0, 0.2, p_num)
    return x, y

def draw_curve(x, y, w, order):
    def draw_sin_curve():
        x = np.linspace(0, 1, 50)
        y = func(x)
        plt.plot(x, y, color='green', linewidth=2)
    
    draw_sin_curve()
        
    def draw_data_scatter(x, y):
        plt.plot(x, y, 'wo')
        
    draw_data_scatter(x, y)
    
    def draw_fit_curve(w, order):
        x = np.linspace(0, 1, 50)
        P_x = np.array([[xi ** i for i in range(order+1)] for xi in x])
        y = P_x.dot(w)
        plt.plot(x, y, color='red', linewidth=2)
        
    draw_fit_curve(w, order)
    
    def curve_setting():
        xlims = plt.xlim()
        ylims = plt.ylim()
        plt.xlim(xlims[0]-0.1, xlims[1]+0.1)
        plt.ylim(ylims[0]-0.1, ylims[1]+0.1)
        plt.title('Polynomial Curve Fitting')
        plt.xlabel('x')
        plt.ylabel('y', rotation='horizontal')
    
    curve_setting()
    plt.show()

def polynomial_fit(x, y, order):
    P_x = np.array([[xi ** i for i in range(order+1)] for xi in x])
    Y = np.array(y).reshape((-1, 1))
    print (P_x.dtype)
    w = np.linalg.inv(P_x.T.dot(P_x)).dot(P_x.T).dot(Y)
    print (w)
    return w
    
def polynomial_fit_with_penality(x, y, order, lam):
    P_x = np.array([[xi ** i for i in range(order+1)] for xi in x])
    Y = np.array(y).reshape((-1, 1))
    w = np.linalg.inv(P_x.T.dot(P_x) + lam * np.eye(p_num)).dot(P_x.T).dot(Y)
    print (w)
    return w

if __name__ == '__main__':
    p_num = 10
    order = 9
    lam = np.e ** (-18)
    
    x, y = gen_data(p_num)
    w = polynomial_fit(x, y, order)
    plt.text(0.7, 1, 'M = '+np.str(order) , fontsize=12)
    draw_curve(x, y, w, order)
    w_penality = polynomial_fit_with_penality(x, y, order, lam)
    plt.text(0.7, 1, r'$\mathrm{ln}\lambda = \mathrm{-18}$', fontsize=12)
    draw_curve(x, y, w_penality, order)

