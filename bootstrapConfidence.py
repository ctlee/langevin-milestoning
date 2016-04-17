#!/bin/env python2.7
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math
import plottools
from bootstrap import bootstrapRho, getRho, bootstrapSKRho
import scikits.bootstrap as skbootstrap


stride = 50
high = 1000

fig = plt.figure(1, facecolor='white', figsize=(7,5.6))

trueProb = 0.5
xs = np.arange(stride, high+stride, stride)
ys = np.ones_like(xs)*trueProb
yerr_low = np.zeros_like(xs, dtype=np.float)
yerr_high = np.zeros_like(xs, dtype=np.float)
i = 0
for x in xs:
    crossPos = x*trueProb
    crossNeg = x-crossPos
    transitions = np.concatenate([np.ones(crossPos), np.ones(crossNeg)*-1],
            axis=0)
    CI = skbootstrap.ci(data=transitions, statfunction=getRho, 
            output='errorbar', n_samples=10000, method='pi') 
    print x, CI
    yerr_low[i] = CI[0,0]
    yerr_high[i] = CI[1,0]
    i+=1

ax1 = fig.add_subplot(111)
ax1.margins(0,0.05)
ax1.errorbar(xs, ys, yerr=[yerr_low, yerr_high], ecolor='r', 
        color='k', fmt='o', elinewidth=2, capthick=2)
ax1.set_ylabel(r'Probability')
ax1.set_xlabel(r'Number of Samples')
ax1.set_xlim([0,high])
ax1.set_ylim([0,1])


"""
trueProb = 0.05
xs = np.arange(stride, high+stride, stride)
ys = np.ones_like(xs)*trueProb
yerr_low = np.zeros_like(xs, dtype=np.float)
yerr_high = np.zeros_like(xs, dtype=np.float)
i = 0
for x in xs:
    crossPos = x*trueProb
    crossNeg = x-crossPos
    transitions = np.concatenate([np.ones(crossPos), np.ones(crossNeg)*-1],
            axis=0)
    CI = skbootstrap.ci(data=transitions, statfunction=getRho,
            output='errorbar', n_samples=10000, method='pi') 
    yerr_low[i] = CI[0,0]
    yerr_high[i] = CI[1,0]
    i+=1

ax2 = fig.add_subplot(212)
ax2.margins(0,0.05)
ax2.errorbar(xs, ys, yerr=[yerr_low, yerr_high], ecolor='r', 
        color='k', fmt='o', elinewidth=2, capthick=2)
ax2.set_ylabel(r'Probability')
ax2.set_xlabel(r'Number of Samples')
ax2.set_xlim([0,high])
ax2.set_ylim([0,0.5])
"""

fig.savefig('figures/bootRho.png', dpi=300)
plt.show()
