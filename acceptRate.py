#!/bin/env python2.7
"""
@description: Performs a 1D Langevin dynamics simulation to validate the results
        of the milestoning permeability derivation
@authors:   Christopher T. Lee (ctlee@ucsd.edu)
@copyright Amaro Lab 2015. All rights reserved.
"""
import argparse, logging, matplotlib, os, pdb
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle 

# Import custom modules
import plottools
import membranesystem as ms
import permeability as perm
from samplefunctions import PMF, Viscosity

if __name__ == '__main__':
    milestones = np.arange(-25, 26, 1)
   
    # Flat system
    pmf = PMF(25, 0, 0, 0, 21, 13)
    nu = Viscosity(25, 3.21e-14, 3.21e-14, 3.21e-14, 3.21e-14, 18, 10, 6)
    system = ms.MembraneSystem('test', pmf, nu, 20, 5, step = 0.01)

    acceptNum = 0
    rejectNum = 0
    forward = 0
    reverse = 0
    fhpdMembers = list()
    for i in np.arange(10000):
        m0, m1, accept, v0, time = system.milestoneC(length = 1e-8,
                dt = 1e-15,
                pos = 0,
                vel = None,
                minx = -1,
                maxx = 1,
                phase = 'reverse',
                reflecting = False)
        if accept:
            fhpdMembers.append(-v0)
            acceptNum += 1
            if m1 == -1:
                reverse += 1
            elif m1 == 1:
                forward += 1
        else:
            rejectNum +=1
    print "Reject: %d, Accept: %d"%(rejectNum, acceptNum)
    print "Reverse: %d, Forward: %d"%(reverse, forward)

    fig = plt.figure(1, facecolor='white',figsize=(7,7))
    ax1 = fig.add_subplot(111)
    ax1.margins(0, 0.05)
    
    count, bins, ignored = plt.hist(fhpdMembers, 100, normed = True)
    ax1.set_title('FHPD')

    plt.show()

