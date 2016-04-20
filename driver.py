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
    # parse the arguments
    # TODO: add more arguments for arbitrary system definition and run
    parser = argparse.ArgumentParser(description="Runs 1-Dimensional " +
            "permeability calculations using a Langevin integrator. Both " +
            "non-milestoned and milestoned permeability values are computed.")
    parser.add_argument('-c',
            "--copt",
            action = 'store_true',
            dest = 'cOpt',
            help="Use the faster C integrator")
    parser.add_argument('-v',
            action='count',
            dest='verbosity',
            help = 'Set the output verbosity')
    args = parser.parse_args() # parse the args into a dictionary

    if args.verbosity == 0:
        logging.basicConfig(level=logging.ERROR, 
            format='[%(levelname)s] %(process)d %(processName)s: %(message)s')
    elif args.verbosity == 1:
        logging.basicConfig(level=logging.WARNING, 
            format='[%(levelname)s] %(process)d %(processName)s: %(message)s')
    elif args.verbosity == 2:
        logging.basicConfig(level=logging.INFO, 
            format='[%(levelname)s] %(process)d %(processName)s: %(message)s')
    elif args.verbosity >= 3:
        logging.basicConfig(level=logging.DEBUG, 
            format='[%(levelname)s] %(process)d %(processName)s: %(message)s')
    cOpt = args.cOpt

    if cOpt == False:
        raw_input("Using non-accelerated, hit enter to continue...")

    milestones = np.arange(-25, 26, 1)
   
    # Flat system
    print '#'*30 + '\n#         Flat System        #\n'  + '#'*30
    name = 'flat'
    if not os.path.exists('systems/%s_system.p'%(name)):
        pmf = PMF(25, 0, 0, 0, 21, 13)
        nu = Viscosity(25, 3.21e-14, 3.21e-14, 3.21e-14, 3.21e-14, 18, 10, 6)
        system = ms.MembraneSystem(name, pmf, nu, 20, 5, step = 0.01)
        pickle.dump(system, open('systems/%s_system.p'%(name), 'wb'), -1)
    else:
        system = pickle.load(open('systems/%s_system.p'%(name), 'rb'))

    fig1, fig2 = system.plotProfiles()
    fig1.savefig('figures/%s_pmf.png'%(name), dpi=300)
    fig2.savefig('figures/%s_dz.png'%(name), dpi=300)
    #plt.show()
    ihsdP = system.inhomogenousSolDiff()
    smolP = system.smolPerm()
    if abs(ihsdP - smolP) > 1e-4:
        logging.warning("Equation 22 is not equal")
    perm.bruteMPCrossing(system, numSims = 1000, length = 1e-6, 
            dt = 1e-12, cOpt = cOpt)
    perm.bruteMPTimes(system, numSims = 1000, length = 1e-6, 
            dt = 1e-12, cOpt = cOpt)
    #perm.processBrute(system)
    
    perm.milestoneMP(system, milestones, numSims = 10000, 
            length = 1e-6, dt = 1e-15, cOpt = cOpt)
    #perm.processMilestones(system, milestones)
    print '\n\n'

    # Small hill system
    print '#'*30 + '\n#      Small Hill System        #\n'  + '#'*30
    name = 'smallhill'
    if not os.path.exists('systems/%s_system.p'%(name)):
        pmf = PMF(25, 0, 0.25, 0.5, 21, 13)
        nu = Viscosity(25, 3.21e-14, 5e-13, 5e-13, 5e-13, 18, 10, 6)
        system = ms.MembraneSystem(name, pmf, nu, 20, 5, step = 0.01)
        pickle.dump(system, open('systems/%s_system.p'%(name), 'wb'), -1)
    else:
        system = pickle.load(open('systems/%s_system.p'%(name), 'rb'))
   
    fig1, fig2 = system.plotProfiles()
    fig1.savefig('figures/%s_pmf.png'%(name), dpi=300)
    fig2.savefig('figures/%s_dz.png'%(name), dpi=300)
    fig1.clear()
    fig2.clear()
    
    ihsdP = system.inhomogenousSolDiff()
    smolP = system.smolPerm()
    if abs(ihsdP - smolP) > 1e-4:
        logging.warning("Equation 22 is not equal")
    """
    perm.bruteMPCrossing(system, numSims=1000, length = 1e-4, 
            dt = 1e-12, cOpt = cOpt)
    perm.bruteMPTimes(system, numSims=500, length = 1e-4, 
           dt = 1e-12, cOpt = cOpt)
    """
    #perm.processBrute(system)

    perm.milestoneMP(system, milestones, numSims = 10000, 
            length = 1e-4, dt = 1e-15, cOpt = cOpt)
    #perm.processMilestones(system, milestones)
    print '\n\n'

    # Urea like system
    print '#'*30 + '\n#         Urea System        #\n'  + '#'*30
    name = 'urea'
    if not os.path.exists('systems/%s_system.p'%(name)):
        pmf = PMF(25, 0, 0.694785, 5.558286, 21, 13)
        nu = Viscosity(25, 3.21e-14, 5e-13, 5e-13, 5e-13, 18, 10, 6)
        system = ms.MembraneSystem(name, pmf, nu, 60, 2.86, step = 0.01)
        pickle.dump(system, open('systems/%s_system.p'%(name), 'wb'), -1)
    else:
        system = pickle.load(open('systems/%s_system.p'%(name), 'rb'))
   
    fig1, fig2 = system.plotProfiles()
    fig1.savefig('figures/%s_pmf.png'%(name), dpi=300)
    fig2.savefig('figures/%s_dz.png'%(name), dpi=300)
    fig1.clear()
    fig2.clear()
    
    ihsdP = system.inhomogenousSolDiff()
    smolP = system.smolPerm()
    if abs(ihsdP - smolP) > 1e-4:
        logging.warning("Equation 22 is not equal")
    """ 
    perm.bruteMPCrossing(system, numSims=1000, length = 1, 
            dt = 1e-12, cOpt = cOpt)
    perm.bruteMPTimes(system, numSims=500, length = 1, 
            dt = 1e-12, cOpt = cOpt)
    """
    #perm.processBrute(system)

    milestones = np.arange(-25, 26, 1)
    perm.milestoneMP(system, milestones, numSims = 10000, 
            length = 1, dt = 1e-15, cOpt = cOpt)
    #perm.processMilestones(system, milestones)
    print '\n\n'

    # codeine system
    print '#'*30 + '\n#      Codeine System        #\n'  + '#'*30
    name = 'codeine'
    if not os.path.exists('systems/%s_system.p'%(name)):
        pmf = PMF(25, 0.25, -2, -0.5, 21, 13)
        nu = Viscosity(25, 3.21e-14, 5e-13, 5e-13, 5e-13, 18, 10, 6)
        system = ms.MembraneSystem(name, pmf, nu, 299, 4.315, step = 0.01)
        pickle.dump(system, open('systems/%s_system.p'%(name), 'wb'), -1)
    else:
        system = pickle.load(open('systems/%s_system.p'%(name), 'rb'))
   
    fig1, fig2 = system.plotProfiles()
    fig1.savefig('figures/%s_pmf.png'%(name), dpi=300)
    fig2.savefig('figures/%s_dz.png'%(name), dpi=300)
    fig1.clear()
    fig2.clear()
    
    ihsdP = system.inhomogenousSolDiff()
    smolP = system.smolPerm()
    if abs(ihsdP - smolP) > 1e-4:
        logging.warning("Equation 22 is not equal")
    """
    perm.bruteMPCrossing(system, numSims=1000, length = 1e-4, 
            dt = 1e-12, cOpt = cOpt)
    perm.bruteMPTimes(system, numSims=500, length = 1e-4, 
            dt = 1e-12, cOpt = cOpt)
    """
    #perm.processBrute(system)

    perm.milestoneMP(system, milestones, numSims = 10000, 
            length = 1e-4, dt = 1e-15, cOpt = cOpt)
    #perm.processMilestones(system, milestones)
    print '\n\n'
