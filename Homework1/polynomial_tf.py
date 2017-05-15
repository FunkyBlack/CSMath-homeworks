# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:54:27 2017

@author: FunkyBlack
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def gen_data(p_num):
    x = np.reshape(np.linspace(0, 1, p_num), (p_num, 1))
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, p_num).reshape((p_num, 1))
    return x, y
    
def weight_variable(shape): 
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='W', trainable=True)
    
def draw_curve(x, y):
    def draw_sin_curve():
        x = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * x)
        plt.plot(x, y, color='green', linewidth=2)
    
    draw_sin_curve()
        
    def draw_data_scatter(x, y):
        plt.plot(x, y, 'bo')
        
    draw_data_scatter(x, y)
  
    def curve_setting():
        xlims = plt.xlim()
        ylims = plt.ylim()
        plt.xlim(xlims[0]-0.1, xlims[1]+0.1)
        plt.ylim(ylims[0]-0.2, ylims[1]+0.2)
        plt.title('Polynomial Curve Fitting Using Tensorflow')
        plt.xlabel('x')
        plt.ylabel('y', rotation='horizontal')
    
    curve_setting()
    
def draw_fit_curve(w, order):
    x = np.linspace(0, 1, 50)
    P_x = np.array([[xi ** i for i in range(order+1)] for xi in x])
    y = P_x.dot(w)
    plt.plot(x, y, color='red', linewidth=2)
    plt.show()
    
def main():
    sess = tf.InteractiveSession()
    
    p_num = 100
    order = 9
    lamb = np.e ** (-18)
    
    x = tf.placeholder(tf.float32, shape=[p_num, 1])
    y_ = tf.placeholder(tf.float32, shape=[p_num, 1])
   
    w = weight_variable([order+1, 1])
    
    P_x = tf.reshape(tf.stack([tf.pow(x, i) for i in range(order+1)], axis=1), (p_num, order+1))
    y = tf.matmul(P_x, w)
    
    loss = 0.5 * tf.reduce_sum(tf.pow(y_-y, 2))
    loss_with_penality = loss + 0.5 * lamb * tf.reduce_sum(tf.pow(w, 2))
    
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    batch = gen_data(p_num)
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 1000 == 0:
            print("loss: ", sess.run(loss_with_penality, feed_dict={x: batch[0], y_: batch[1]}))
    
    w_value = w.eval()
    draw_curve(batch[0], batch[1])
    draw_fit_curve(w_value, order)
    plt.savefig('CurveFittingTF_num{}.png'.format(str(p_num)))
    
if __name__ == '__main__':
    main()

    
    