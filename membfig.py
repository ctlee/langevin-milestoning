#!/bin/env python
"""
@description: Performs a 1D Langevin dynamics simulation to validate the results
        of the milestoning pemreability derivation
@authors:   Christopher T. Lee (ctlee@ucsd.edu)
@copyright Amaro Lab 2015. All rights reserved.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plottools
        
fig = plt.figure(1, facecolor='white', figsize=(7, 5.6))
ax1 = fig.add_subplot(111)

a = 2
b = 8
x = [0,a,b,10]
y = [1,1,0,0]

ax1.plot(x, y, lw=3, color='r')
ax1.axvline(x=a, lw=2, color='k', linestyle='--')
ax1.axvline(x=b, lw=2, color='k', linestyle='--')
ax1.set_ylabel(r'Concentration [$u(z)$]')
ax1.set_xlabel(r'Position [$z$]')

ylabels = [0, r'$u_0$']
plt.yticks([0,1], ylabels)

xlabels = [0, 'a', 'b']
plt.xticks([0,a,b], xlabels)

ax1.text(0.25,0.75, 'Donor Compartment', rotation='vertical', )
ax1.text(9.5,0.76, 'Acceptor Compartment', rotation='vertical', )
ax1.margins(0,0.05)
plt.savefig('figures/ISDplot.png', dpi=300, transparent=True)



fig = plt.figure(2, facecolor='white', figsize=(7, 5.6))
ax1 = fig.add_subplot(111)

q = 1.5
a = 5
b = 10
x = [0,q,a]
y = [1,0.95,0]

ax1.plot(x, y, lw=3, color='r')
ax1.axvline(x=0, lw=2, color='k', linestyle='--')
ax1.axvline(x=q, lw=2, color='k', linestyle='--')
ax1.axvline(x=a, lw=2, color='k', linestyle='--')
ax1.axvline(x=b, lw=2, color='k', linestyle='--')

for i in xrange(a,b,1):
    ax1.axvline(x=i, lw=0.5, color='k', linestyle='--')
ax1.set_ylabel(r'Concentration [$u(z)$]')
ax1.set_xlabel(r'Position [$z$]')

ylabels = [0, r'$u_0$']
plt.yticks([0,1], ylabels)

xlabels = [0, 'q', 'b', 'a']
plt.xticks([0,q,a,b], xlabels)

ax1.set_ylim([0,2])
ax1.margins(0.05,0.05)

ax1.text(-0.4,1.5, 'Donor Compartment', rotation='vertical', )
ax1.text(10.1,1.55, 'Acceptor Compartment', rotation='vertical', )




ax1.text(0.5,1.15, r'$J_1$')
ax1.arrow(0.1, 1.1, 1.2, 0, head_length=0.1, head_width=0.05, 
        width=0.015, facecolor='green')

ax1.text(3,0.65, r'$J_2$')
ax1.arrow(1.7, 0.6, 3, 0, head_length=0.15, head_width=0.06,
        width=0.02, facecolor='green')


ax1.text(7.5, 1.05, r'$\mathbf{\rho}$', fontsize=20, color='blue', 
        bbox=dict(facecolor='white', alpha=0.85, 
                joinstyle='round', linewidth=0))
ax1.arrow(5.2, 0.95, 4.6, 0, head_length=0.1, head_width=0.05,
        width=0.015, facecolor='blue')

ax1.text(2.9, 1.05, r'$\mathbf{1-\rho}$', fontsize=20, color='blue')
ax1.arrow(4.8, 0.95, -3.1, 0, head_length=0.1, head_width=0.05,
        width=0.015, facecolor='blue')

ax1.text(6.75,0.05, r'Milestones')

plt.savefig('figures/probplot.png', dpi=300, transparent=True)
plt.show()
