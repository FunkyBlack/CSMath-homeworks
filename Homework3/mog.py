# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:07:28 2017

@author: FunkyBlack
"""

import numpy as np
import matplotlib.pyplot as plt
#==============================================================================
# from matplotlib.legend_handler import HandlerPatch
# import matplotlib.patches as mpatches
#==============================================================================
from matplotlib.patches import Ellipse

np.random.seed(1)
color_arr=['red','green','blue','yellow','white','black','cyan','magenta']

#==============================================================================
# class HandlerEllipse(HandlerPatch):
#     def create_artists(self, legend, orig_handle,
#                        xdescent, ydescent, width, height, fontsize, trans):
#         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
#         p = mpatches.Ellipse(xy=center, width=width + xdescent,
#                              height=height + ydescent)
#         self.update_prop(p, orig_handle, legend)
#         p.set_transform(trans)
#         return [p]
#==============================================================================

def BoxMullerSampling(mu=[0,0], sigma=[1,1], size=(1000,2)):
    u1 = np.random.uniform(size=size)
    u2 = np.random.uniform(size=size)
    x = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z = np.dot(x, np.diag(sigma)) + mu
    return z

def MixtureOfGaussian(ndim=4, mu=[[0,0],[5,5],[2,3],[8,8]], sigma=[[1,1],[0.5,0.5],[1,2],[0.5,1]], size=600): # Given the parameters of MOG
    data = {}
    sample = np.random.randint(low=0, high=size, size=ndim)
    p = sample / np.sum(sample) # Determine the mixture density
    n_samples = p * size
    n_samples = map(lambda x: int(x), n_samples)
    plt.figure(0)
    for i, n_sample in enumerate(n_samples):
        data[i] = BoxMullerSampling(mu=mu[i],sigma=sigma[i],size=(n_sample,2))
        plt.plot(data[i][:,0],data[i][:,1],'o',color='{}'.format(color_arr[i]), markersize=2.0)

    ells = [Ellipse(xy=mu[j], width=6 * sigma[j][0], height=6 * sigma[j][1]) for j in range(ndim)]
    fig = plt.figure(0, figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('MOG')
    for j,e in enumerate(ells):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor('none')
        e.set_edgecolor('{}'.format(color_arr[j]))
    ax.set_xlim(-4, 12)
    ax.set_ylim(-4, 12)    
    
    plt.show()
    
    data_gen = np.concatenate([data[i] for i in range(ndim)], axis=0)
    return p, data_gen
    
def E_step(data, ndim=2, p_sample=[0.5,0.5], mu=[[1,1],[3,3]], sigma=[[[9,0],[0,9]],[[9,0],[0,9]]]):
    n_samples = data.shape[0]
    p_estimate = np.zeros(shape=(ndim, n_samples))
    p_estimate_sum = np.zeros(shape=(1, n_samples))
    for j in range(n_samples):
        for i in range(ndim):
            p_gaussian = 1 / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(sigma[i])) * np.exp(-0.5 * np.dot(np.dot((data[j] - mu[i]), np.linalg.inv(sigma[i])), (data[j] - mu[i]).T))
            p_estimate[i][j] = p_sample[i] * p_gaussian
            p_estimate_sum[:,j] += p_estimate[i][j]
    p_estimate = p_estimate / p_estimate_sum
    
    return p_estimate
    
def M_step(data, p_estimate=None):
    
    n_samples = data.shape[0]
    ndim = p_estimate.shape[0]
    if p_estimate == None:
        p_estimate = np.zeros(shape=(ndim, n_samples))

    mu_estimate_sum = [0 for i in range(ndim)]
    mu_estimate = [0 for i in range(ndim)]
    sigma_estimate_sum = [0 for i in range(ndim)]
    sigma_estimate = [0 for i in range(ndim)]
    n_estimate = np.sum(p_estimate, axis=1)
    p_sample_estimate = n_estimate / n_samples # Mixture density we obtain from EM
    for i in range(ndim):
        for j in range(n_samples):
            mu_estimate_sum[i] += p_estimate[i][j] * data[j]
        mu_estimate[i] = mu_estimate_sum[i] / n_estimate[i] # Mu we obtain from EM
    for i in range(ndim):
        sigma_estimate_sum[i] = np.zeros(shape=(2,2))
        for j in range(n_samples):
            sigma_estimate_sum[i] += p_estimate[i][j] * np.dot((data[j] - mu_estimate[i]).reshape(-1,1), (data[j] - mu_estimate[i]).reshape(1,-1))
        sigma_estimate[i] = sigma_estimate_sum[i] / n_estimate[i]

    return p_sample_estimate, mu_estimate, sigma_estimate
    
def EM_iteration(data):
    Max_iter = 20
    epsilon = 1e-1
    
    ####  Initialization ####
    ndim=4
    p_sample=[0.25,0.25,0.25,0.25]
    mu=[[1,1],[3,3],[2,2],[6,6]]
    sigma=[[[2,0],[0,2]],[[5,0],[0,5]],[[3,0],[0,3]],[[4,0],[0,4]]]
    ####
    
    n_samples = data.shape[0]
    p_estimate = np.zeros(shape=(ndim, n_samples))
    for i in range(Max_iter):
        
        ells = [Ellipse(xy=mu[j], width=6 * np.sqrt(sigma[j][0][0]), height=6 * np.sqrt(sigma[j][1][1])) for j in range(ndim)]
        fig = plt.figure(1, figsize=(18,10))
        ax = fig.add_subplot(4,4,i+1)
        ax.set_title('Iteration: {}'.format(i+1))
        for j,e in enumerate(ells):
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor('none')
            e.set_edgecolor('{}'.format(color_arr[j]))
        ax.set_xlim(-4, 12)
        ax.set_ylim(-4, 12)
        plt.plot(data[:,0],data[:,1],'o',color='{}'.format(color_arr[-2]), markersize=0.9)
        plt.show()
        plt.savefig('EM_iterations_ndim4.png')
        
        p_estimate = E_step(data=data, ndim=ndim, p_sample=p_sample, mu=mu, sigma=sigma)
        old_mu = mu
        p_sample, mu, sigma = M_step(data=data, p_estimate=p_estimate)
        if(np.sum(np.sum(np.abs(np.array(old_mu) - np.array(mu)))) < epsilon):
            print ("Iterations: ", i)
            ells = [Ellipse(xy=mu[j], width=6 * np.sqrt(sigma[j][0][0]), height=6 * np.sqrt(sigma[j][1][1]), linestyle='dashed') for j in range(ndim)]
            fig = plt.figure(0, figsize=(10,5))
            ax = fig.add_subplot(1,1,1)
            for j,e in enumerate(ells):
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(1)
                e.set_facecolor('none')
                e.set_edgecolor('{}'.format(color_arr[j]))
            ax.set_xlim(-4, 12)
            ax.set_ylim(-4, 12)
            plt.show()
            plt.savefig('MOG_Using_EM_ndim4.png')
            break
    
    return p_sample, mu, sigma
    
if __name__ == '__main__':
    p_sample, data_gen = MixtureOfGaussian(size=600)
    print ("p_sample: ", p_sample)
    print ("num_samples: ", data_gen.shape[0])
    
    p_sample_estimate, mu, sigma = EM_iteration(data_gen)
    print (p_sample_estimate, mu, sigma)