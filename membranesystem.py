import logging, sys
import numpy as np
from math import exp, log, pi, sqrt, ceil

import matplotlib
if sys.platform == 'darwin':
    matplotlib.use('macosx') # cocoa rendering for Mac OS X
import matplotlib.pyplot as plt

import cPickle as pickle
import traj_tools
from samplefunctions import PMF, Viscosity

# Global Constants
kb = 0.0013806488   # kgA^2s^-2K-1
avogadro = 6.022e23     # Avogradros number

class MembraneSystem():
    def __init__(self, name, pmf, viscosity, mass, r, 
            T = 298, dz = 25, step = 0.0001):
        self.name = name
        self.pmf = pmf
        self.viscosity = viscosity
        self.step = step
        self.z = np.arange(-dz, dz+step, step)
        self.T = T
        self.m = mass*1e-3/6.022e23 # convert from g/mol to kg
        self.r = r
        self.dz = abs(dz)

    def plotProfiles(self):
        fig1 = self.pmf.plot(1)
        fig2 = plt.figure(2, facecolor='white', figsize=(7, 5.6))
        ax1 = fig2.add_subplot(111)
        ax1.margins(0, 0.05)

        diffusivity = self.getD(self.z)*1e-10   # A^2/s -> 1e6 cm^2/s
        ax1.plot(self.z, diffusivity)
        ax1.set_ylabel(r'Diffusivity [$cm^2/s \times 10^{-6}$]')
        ax1.set_xlabel(r'Position [$\AA$]')
        return fig1, fig2

    def getD(self, x):  # viscosity in units of kg/A/s
        global kb
        return kb*self.T/(6*pi*self.viscosity(x)*self.r)   # A^2/s

    def inhomogenousSolDiff(self):
        """
        Calculate the inhomogenous solubility diffusion result
        """
        global kb
        resistivity = np.trapz(
                np.exp(self.pmf(self.z)/(kb*self.T))
                / self.getD(self.z), self.z)    # s/A
        P = 1/resistivity   # A/s
        Pcm = P*1e-8    # cm/s
        print("Inhomogenous Solubility Diffusion Equation Permeability: " +
            "%e cm/s; log(P) %f"%(Pcm, log(Pcm, 10)))
        return log(Pcm, 10)

    def smolPerm(self):
        global kb, avogadro
        # Calculate the mfpt by eq. 19...
        # Integral of e^(-W(z))/kT
        y1 = np.exp(-self.pmf(self.z)/(kb*self.T))
       
        # TODO: resolve which y2 we should be using...
        y2 = [np.trapz(
                np.exp(
                    self.pmf(np.arange(-self.dz, z, self.step)) /
                    (kb*self.T)
                ) /
                self.getD(np.arange(-self.dz, z, self.step)),
                dx = self.step)
                for z in self.z]

        """
        y2 = [np.trapz(
                np.exp(
                    self.pmf(np.arange(-self.dz, z+self.step, self.step)) /
                    (kb*self.T)
                ) /
                self.getD(np.arange(-self.dz, z+self.step, self.step)),
                dx = self.step)
                for z in self.z]
        """
        y = np.multiply(y1,y2)  # Elementwise product

        mfpt = np.trapz(y, self.z)
        print("Eq 22 MFPT: %e s"%(mfpt)) # s
        P = np.trapz(y1, dx = self.step)/(2*mfpt)   # A/s
        Pcm = P*1e-8    # cm/s
        print("Eq 22 Permeability: %e cm/s; log(P) %f"%(Pcm, log(Pcm,10)))
        return log(Pcm, 10)
   
    def cumulativeProbDist(self):
        global kb, avogadro
        return np.sum(np.exp(-self.pmf(self.z)/(kb*self.T)))*self.step

    def milestoneC(self,
            length = 2e-8,              # Length (s)
            dt = 2e-15,                 # dt (s)
            pos = 0,                    # Initial position (A) 
            vel = None,                    # initial velocity (A/s)
            minx = -1.0,                 # lower boundary
            maxx = 1.0,                 # upper boundary
            phase = "forward",
            reflecting = False):          # which phase of calculation we are in
        
        logging.debug("Starting milestone C")
        if pos > maxx or pos < minx:
            raise ValueError("Position %f not in bounds of min/max (%f, %f)"
                %(pos, minx, maxx))
        N = int(ceil(length/dt))
        if vel == None: # sample from a Maxwell-Boltzmann distribution
            vel = np.random.normal(0.0, sqrt(kb*self.T/self.m))
       
        # Cast to c happy integer booleans, instead of mucky PyObjects
        if phase == "forward":
            reverse = 0
        else:
            reverse = 1
        if reflecting:
            refecting = 1
        else:
            reflecting = 0  
        accept, finalpos, time = traj_tools.milestoneO(self.pmf.force, 
                self.viscosity, self.m, self.r, self.T, pos, vel, minx, 
                maxx, dt, N, reverse, reflecting)
        time = dt*time
        logging.debug("Accept: %d; finalpos: %f; time: %e"
                %(accept, finalpos, time))
        accept = True if accept == 1 else False
        return pos, finalpos, accept, vel, time

    def milestone(self,
            length = 2e-8,              # Length (s)
            dt = 2e-15,                 # dt (s)
            pos = 0,                    # Initial position (A)
            vel = None,                 # initial velocity (A/s)
            minx = -1.0,                # lower boundary
            maxx = 1.0,                 # upper boundary
            phase = "forward",
            reflecting = False):        # which phase of calculation we are in
        """
        Generate a sample trajectory for later analysis
        """
        global kb
        if pos > maxx or pos < minx:
            raise ValueError("Position %f not in bounds of min/max (%f, %f)"
                    %(pos, minx, maxx))
        
        N = int(ceil(length/dt))   # Max number of trajectory steps
        if vel == None: # sample from a Maxwell-Boltzmann distribution
            # stdev of a velocity distribution
            vel = np.random.normal(0.0, sqrt(kb*self.T/self.m))
        startvel = vel
        startpos = pos
        oldpos = startpos
        force = self.pmf.force(pos) # Initialize the starting force
        
        for i in xrange(1,N):       # interate steps
            c = 6*pi*self.viscosity(pos)*self.r # position dependent damping 
                                                # coefficient from Stokes' Law
                                                # kg*s^-1
            b = 1/(1+c*dt/(2*self.m))           # unitless
            a = (1 - c*dt/(2*self.m))/(1 + c*dt/(2*self.m)) # unitless
            noise = sqrt(2*c*kb*self.T*dt)  # Fluctuation-Dissipation Theorem
                                            # (kg*A*s^-1)
            bump = np.random.normal(0,noise)    # Generate the random kick 
                                                # with variance noise
            """
            This integrator is an implementation of:
            Gronbech-Jensen, N., Farago, O., Molecular Physics, V111, 8:989-991,
            (2013)
            """
            pos = pos + b*dt*vel + b*dt*dt*force/(2*self.m) + \
                    b*dt/(2*self.m)*bump                        # A  
            fnew = self.pmf.force(pos)                          # kg*A*s^-2
            vel = a*vel + dt/(2*self.m)*(a*force + fnew) + \
                    b/self.m*bump                               # A/s
            force = fnew
            #print "step: %d position: %e"%(i, pos)
            # if we go out of bounds, then cancel the simulation
            if phase == "reverse" and i > 1:
                # if self crossing then reject
                if (pos - startpos > 0) != (oldpos - startpos > 0):
                    logging.debug("Accept: %d; finalpos: %f; time: %e"
                            %(0, startpos, dt*i))
                    return startpos, startpos, False, startvel, dt*i  # reject
            if pos < minx or pos > maxx:
                # Reflect if less than min...
                if reflecting == True and pos < minx:
                    pos = minx + (minx - pos)
                    vel = -vel
                elif pos < minx:
                    logging.debug("Accept: %d; finalpos: %f; time: %e"
                            %(1, pos, dt*i))
                    return startpos, minx, True, startvel, dt*i   # accept
                elif pos > maxx:
                    logging.debug("Accept: %d; finalpos: %f; time: %e"
                            %(1, pos, dt*i))
                    return startpos, maxx, True, startvel, dt*i   # accept
            oldpos = pos

        logging.debug("Accept: %d; finalpos: %f; time: %e"%(0, pos, -1))
        return startpos, pos, False, startvel, -1 # reject
