"""
@description: Performs a 1D Langevin dynamics simulation to validate the results
        of the milestoning pemreability derivation
@authors:   Christopher T. Lee (ctlee@ucsd.edu)
@copyright Amaro Lab 2015. All rights reserved.
"""
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator as pchip
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use('macosx') # cocoa rendering for Mac OS X

import matplotlib.pyplot as plt
import plottools
#from numba import jit

class PMF(object):
    def __init__(self, dz, w1, w2, w3, a, b):
        """
        Initialize a PMF profile
        :param dz: distance from center to edge of membrane
        :param w1: PMF at interface in units of kgA^2s^-2
        :param w2: PMF at interface/core
        :param w3: PMF at core
        :param a: location of interface (w1)
        :param b: loction of interface/core (w2)
        :type dz: int
        :type w1: float
        :type w2: float
        :type w3: float
        :type a: int
        :type b: int
        """
        # INFO 1kcal = 4184e20 kg A^2 s^-2
        # TODO check bounds of a, b
        self.dz = dz
        self.x = np.array([-dz, -dz+1, -a, -b, 0, b, a, dz-1, dz])
        self.y = np.array([0, 0, w1, w2, w3, w2, w1, 0, 0])
        self.pmf = pchip(self.x, self.y)
        self.derivative = self.pmf.derivative(1)

    #@jit(nogil=True)
    def __call__(self, x, nu = 0):
        return self.pmf(x, nu = nu, extrapolate=True)
    
    #@jit(nogil=True)
    def energy(self, x):
        return self.pmf(x, extrapolate=True)

    #@jit(nogil=True)
    def force(self, x):
        return -self.derivative(x, extrapolate=True)

    def plot(self, fignum = 1):
        scalefactor = 1.43929254302   # conversion to kcal/mol
        extraBound = 5
        
        fig = plt.figure(fignum, facecolor='white', figsize=(7, 5.6))
        ax1 = fig.add_subplot(111)

        x = np.arange(-self.dz-extraBound, self.dz+extraBound+0.1, 0.1)
        pmf = self.energy(x)*scalefactor
        
        ax1.plot(self.x, self.y*scalefactor, 'o', label='Supplied Values')
        ax1.plot(x, pmf, label='Spline of PMF')
        ax1.set_ylabel(r'PMF [$kcal/mol$]')
        ax1.set_xlabel(r'Position [$\AA$]')

        ax2 = ax1.twinx()
        force = self.force(x)*scalefactor
        ax2.plot(x, force, label=r'Force [$-dPMF/dx$]')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, 
                loc = 'lower left',
                fontsize = 'small',
                frameon = False)

        ax1.margins(0,0.05)
        ax2.margins(0,0.2)

        lim = [-1,1]
        if ax1.get_ylim()[0] > ax2.get_ylim()[0]:   # Pick the lesser
            lim[0] = ax2.get_ylim()[0]
        else:
            lim[0] = ax1.get_ylim()[0]

        if ax1.get_ylim()[1] > ax2.get_ylim()[1]:
            lim[1] = ax1.get_ylim()[1]
        else:
            lim[1] = ax2.get_ylim()[1]
        ax1.set_ylim(lim)
        ax2.set_ylim(lim)
        return fig
    
class Viscosity(object):
    # TODO document this function
    def __init__(self, dz, d1, d2, d3, d4, a, b, c):
        self.dz = dz
        self.x = [-dz, -a, -b, -c, 0, c, b, a, dz]
        self.y = [d1, d1, d2, d3, d4, d3, d2, d1, d1]
        self.pchip = pchip(self.x, self.y)

    #@jit(nogil=True)
    def __call__(self, x):
        return self.pchip(x, extrapolate = True)

    def plot(self, fignum = 1):
        extraBound = 5
        
        fig = plt.figure(fignum, facecolor='white', figsize=(7,5.6))
        ax1 = fig.add_subplot(111)
        ax1.margins(0,0.05)
        ax1.set_yscale('log')
        x = np.arange(-self.dz-extraBound, self.dz+extraBound+0.1, 0.1)
        y = self.pchip(x)
        ax1.plot(self.x, self.y, 'o', x, y)
        ax1.set_ylabel(r'Viscosity [$kg/\AA\cdot s$]')
        ax1.set_xlabel(r'Position [$\AA$]')
