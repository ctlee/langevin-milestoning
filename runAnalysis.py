#!/bin/env python
import logging, matplotlib, os, sys
import numpy as np
import cPickle as pickle
import permeability as perm
import matplotlib.pyplot as plt

"""
logging.basicConfig(level=logging.INFO, 
        format='[%(levelname)s] %(process)d %(processName)s: %(message)s')
"""
systems = ['flat', 'smallhill', 'urea', 'codeine']

prefix = 'datasets/'
for sys in systems:
    print '#'*30 + '\n#      %s        #\n'%(sys)  + '#'*30
    try:
        system = pickle.load(open('systems/%s_system.p'%(sys), 'rb'))
        milestones = np.load(prefix + sys + '_milestones')
    except:
        print "Could not find system... Continuing"

    fig1, fig2 = system.plotProfiles()
    fig1.savefig('figures/%s_pmf.png'%(sys), dpi=300)
    fig2.savefig('figures/%s_dz.png'%(sys), dpi=300)
    plt.close('all')

    ihsdP = system.inhomogenousSolDiff()
    smolP = system.smolPerm()
    if abs(ihsdP - smolP) > 1e-4:
        logging.warning("Equation 22 is not equal")
    
    perm.processBrute(system, prefix)
    perm.processMilestones(system, milestones, prefix)
    """
    try:
        perm.processBrute(system, prefix)
    except Exception as e:
        print e
    try:
        perm.processMilestones(system, milestones, prefix)
    except Exception as e:
        print e
    """
    print '\n\n'
