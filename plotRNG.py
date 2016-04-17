#!/usr/bin/env python2.7
"""
@description: Plot the gaussian random noise gen
@authors:   Christopher T. Lee (ctlee@ucsd.edu)
@copyright Amaro Lab 2015. All rights reserved.
"""
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plottools, traj_tools

if __name__ == '__main__':
    mu, sigma = 0, 1
    blocks = 100
    dims = 1000
    nums = np.zeros(dims*dims)
    traj_tools.testRNG(nums, dims*dims)    
    fig = plt.figure(1, facecolor='white',figsize=(7,7))
    ax1 = fig.add_subplot(221)
    ax1.margins(0, 0.05)
    
    count, bins, ignored = plt.hist(nums, blocks, normed = True)
    ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')
    ax1.set_title('Binned C Distribution')

    # reshape and plot as a heatmap
    nums = np.reshape(nums, (dims, dims))
    ax2 = fig.add_subplot(222)
    ax2.margins(0, 0.05)
    plt.imshow(nums, interpolation='nearest', cmap=cm.Greys)
    ax2.set_title('C Values')

    nums = np.random.normal(size=dims*dims)
    ax3 = fig.add_subplot(223)
    ax3.margins(0, 0.05)
    
    count, bins, ignored = plt.hist(nums, blocks, normed = True)
    ax3.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')
    ax3.set_title('Binned Numpy Distribution')

    # reshape and plot as a heatmap
    nums = np.reshape(nums, (dims, dims))
    ax4 = fig.add_subplot(224)
    ax4.margins(0, 0.05)
    plt.imshow(nums, interpolation='nearest', cmap=cm.Greys)
    ax4.set_title('Numpy Values')

    fig.subplots_adjust(hspace=0.5)
    fig.savefig('figures/testRNG.png', dpi=300) 
    #plt.show()

