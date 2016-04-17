# Lane Votapka
# Amaro lab 2015
# UCSD
'''
Performs a 1D Langevin dynamics simulation that can be used to validate the results of the milestoning permeability derivation


'''

#import model # import Chris' script
from model import kb
import analyze
import argparse
from markov import monte_carlo_milestoning_nonreversible_error
from markov import rate_mat_to_prob_mat
#import markov
from math import sin, cos, exp, sqrt, pi
import math
import numpy as np
import scipy as sp
import copy
from scipy.fftpack import rfft, irfft
from math import exp, log, sin, cos, pi, sqrt
import matplotlib.pyplot as plt
import time #, random
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

#k = 1.0
a = -1.0
b = 10.0
n = 12
h = (b-a)/(n-1)
print "h:", h

minx = a
maxx = b
end_milestones = [minx, maxx]
milestones = np.arange(a, b+h, h)
avos = 6.022e23


print "Parsing arguments"
# parse the arguments
parser = argparse.ArgumentParser(description="Runs 1-Dimensional permeability calculations using a Langevin integrator. Both non-milestoned and milestoned permeability values are computed.")
parser.add_argument('-k', '--energy', dest="k_factor", default=0.0, type=float, help="Factor which to set the energy height")
parser.add_argument('-r', '--radius', dest="radius", default=100.0, type=float, help="The hydrodynamic radius of the permeant (in A)")
parser.add_argument('-n', '--number', dest="num_sims", default=10000, type=int, help="The number of simulations to run.")

args = parser.parse_args() # parse the args into a dictionary
args = vars(args)
#print args
k_factor = args['k_factor']
hydro_rad = args['radius']
num_sims = args['num_sims']



def trap_rule(func, a, b, h, T, kb, k=1.0):
  ''' use the trapezoid rule to integrate a function '''
  area = 0.0 # warning: unnormalized
  for x in np.arange(a, b, h):
    area += exp(func(x,k,b=b)/(kb*T)) + exp(func(x+h,k,b=b)/(kb*T)) # find the population
    
  return area*h/2.0
  
def estimate_time(energy_func, a, b, h, T, kb, k=1.0):
  area = 0.0 # warning: unnormalized
  for x in np.arange(a, b, h):
    area += exp(-energy_func(x,k, b=b)/(kb*T))*trap_rule(energy_func, a, x, h, T, kb, k=k) + exp(-energy_func(x+h,k,b=b)/(kb*T))*trap_rule(energy_func, a, x+h, h, T, kb, k=k) # find the population
    
  return area*h/2.0

def run_simulation(k,               # energy scaling constant (kg/s^2) # This function is modified from Chris Lee's code
        mass,                           # Mass (kg)
        r,                              # hydrodynamic radius  (A)
        force_func,  # the function that describes the position-dependent force
        energy_func, # the function that describes the position-dependent potential energy
        viscosity = 8.94*math.pow(10,-14),  # kg/A/S of water
        T = 298,                            # Temperature (K) 
        length = 2*math.pow(10, -8),        # Length (s)
        dt = 2*math.pow(10,-15),            # dt (s)
        pos = 0,                            # Initial position (A) 
        vel = None,                         # initial velocity (A/s)
        verbosity = 0,                      # How much to write out
        random = 1,
        minx = 0.0,    # the minimum position before crossing over another milestone
        maxx = 1.0,
        domain_a = 0.0,
        domain_b = 1.0, # the boundaries of the domain for the potential and force evals
        phase = "forward",
        reflecting = False ):  # the maximum position before crossing over another milestone
    """
    Generate a sample trajectory for later analysis
    """
    # Set some initial conditions
    
    if vel == None:
      #vel = 0
      # sample a Maxwell-Boltzmann distribution
      sigma = sqrt(kb*T/mass) # standard deviation of a velocity distribution
      vel = np.random.normal(0.0, sigma)
    
    force = 0
    #k = k*0.694769 # kg*s^-2
    #mass = mass*math.pow(10,-3)/(6.022*math.pow(10,23))  # kg
    c = 6*math.pi*viscosity*r # damping coefficient from Stokes' Law kg*s^-1
    b = 1/(1+c*dt/(2*mass))    # unitless
    a = (1 - c*dt/(2*mass))/(1 + c*dt/(2*mass))   # unitless
    startpos = pos
        
    N = int(math.ceil(length/dt))
    if verbosity >= 1:
      print "Length = " + str(N)
    if verbosity >= 1:
      if k != 0:
        print 'Damping Ratio = ' + str(c/(2*math.sqrt(mass*k)))    # unitless
    noise = math.sqrt(2*c*kb*T*dt)    # Fluctuation-Dissipation Theorem (kg*A*s^-1)
    if verbosity >= 1:
      print "k = " + str(k) + ' kg/s^2'
      print "Mass = " + str(mass)  + " kg"
      print "Temp = " + str(T) + " K"
      print "r = " + str(r) + " A"
      print "Noise = " + str(noise) + " kgA/s"
      print "Damping Coefficient = " + str(c) + " kg/s"

    if not random:
      np.random.seed(12345)   # my favorite number
    noises = np.random.normal(0, noise, size=N) 
    
    # Preallocate the arrays for the results 
    positions = np.zeros(N, dtype=np.float64)
    #noises = np.empty(N, dtype=np.float64)
    forces = np.zeros(N, dtype=np.float64)
    KE = np.zeros(N, dtype=np.float64)
    PE = np.zeros(N, dtype=np.float64)
    energies = np.zeros(N, dtype=np.float64)
    velocities = np.zeros(N, dtype=np.float64)
    oldpos = startpos
    force = force_func(pos, k, a=domain_a, b=domain_b)
    for i in xrange(0,N):   # interate steps
      positions[i] = pos
      #bump = random.gauss(0,noise)    # Generate the random kick with variance noise
      #noises[i] = bump
      """
      This integrator is an implementation of:
      Gronbech-Jensen, N., Farago, O., Molecular Physics, V111, 8:989-991, (2013)
      """
      
      pos = pos + b*dt*vel + b*dt*dt*force/(2*mass) + b*dt/(2*mass)*noises[i]    # A
      fnew = force_func(pos, k,a=domain_a, b=domain_b) #-k*pos + bias    # harmonic oscillator + underlying PMF
      forces[i] = fnew        # kg*A*s^-2
      vel = a*vel + dt/(2*mass)*(a*force + fnew) + b/mass*noises[i] # A/s
      force = fnew
      velocities[i] = vel
      if verbosity >= 2:
        print "velocity = " + str(vel) + " A/s"
      KE[i] = .5*mass*vel**2
      PE[i] = energy_func(pos, k,a=domain_a, b=domain_b) #.5*k*pos**2
      energies[i] = .5*mass*vel**2 + energy_func(pos, k,a=domain_a, b=domain_b) #.5*k*pos**2 # kgA^2/s^2
      
      # if we go out of bounds, then cancel the sim
      if phase == "reverse" and i > 1: # if there's a reversal phase, then monitor for a transition across the starting position, make sure time has sufficiently passed
        if (pos - startpos > 0) != (oldpos - startpos > 0):# then a crossing even to our own milestone has occurred, reject this point
          #print "rejecting. pos:", pos
          return positions, velocities, forces, noises, KE, PE, energies, pos, dt*i, True # reject it
      
      if pos < minx or pos > maxx:
        #print "Out of bounds...time:", dt*i, " pos:", pos, " not rejecting."
        if reflecting == False or pos > maxx:
          return positions, velocities, forces, noises, KE, PE, energies, pos, dt*i, False # don't reject
        elif reflecting == True and pos < minx:
          #print "reflecting. Oldpos:", pos, "oldvel:", vel
          pos = minx + (minx - pos)
          vel = vel * -1
          #print "new pos:", pos, "new vel:", vel
          
      
      oldpos = pos
    #if verbosity >= 1:
    #  print "===== Done generating sample trajectory! ====="
    
    print "MAX TIME REACHED!"
    return positions, velocities, forces, noises, KE, PE, energies, pos, dt*i, True # reject it


starttime = time.time()
print "Parsing arguments"

# sine curve

def cos1well(x, k=1.0, a=0.0, b=1.0): # a cosine function with one well, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0 or x > b:
    return 0.0
  else: 
    return (cos(2*x*pi/(b))-1)*(0.5*k)
  
def cos1well_force(x, k=1.0, a=0.0, b=1.0): # a cosine function with one well, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0 or x > b:
    #print "OOB"
    return 0.0
  else: 
    #print "x:", x, "k:", k, "a:", a, "b:", b, "force:", k*pi*sin(2*x*pi/(b))
    return (k*pi/b)*sin(2*x*pi/(b))
  
def cos2well(x, k=1.0, a=0.0, b=1.0): # a cosine function with two wells, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0 or x > b:
    return 0.0
  else: 
    return (cos(4*x*pi/(b))-1)*(0.5*k)
  
def cos2well_force(x, k=1.0, a=0.0, b=1.0): # a cosine function with two wells, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0 or x > b:
    return 0.0
  else: 
    return (2*k*pi/b)*sin(4*x*pi/(b))

'''
# unnecessary because we can just invert the constant k
def cos1barrier(x, k=1.0, a=0.0, b=1.0): # a cosine function with one barrier, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0.0 or x > 1.0:
    return 0.0
  else: 
    return -(cos(2*x*pi)-1)*(0.5*k)
  
def cos1barrier_force(x, k=1.0, a=0.0, b=1.0): # a cosine function with one well, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0.0 or x > 1.0:
    return 0.0
  else: 
    return -k*pi*sin(2*x*pi)
  
def cos2barrier(x, k=1.0, a=0.0, b=1.0): # a cosine function with two barrier, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0.0 or x > 1.0:
    return 0.0
  else: 
    return -(cos(4*x*pi)-1)*(0.5*k)
  
def cos2barrier_force(x, k=1.0, a=0.0, b=1.0): # a cosine function with two wells, in a domain between 0 and 1, and a range from -k to 0, to keep it simple
  if x < 0.0 or x > 1.0:
    return 0.0
  else: 
    return -2*k*pi*sin(4*x*pi)
'''
# run a simulation in one of these barriers

length = 2e-5 # s
dt = 1e-12 #s
N = int(math.ceil(length/dt)) # number of timesteps

total_num = num_sims #100000
total_time = 0.0

milestone_trans = {}
milestone_lagtime = {}
for i in range(1,n-1):
  milestone_trans[i] = {i-1:0, i+1:0}
  milestone_lagtime[i] = 0.0
milestone_trans[0] = {1:1e9}
milestone_lagtime[0] = 0.0
milestone_trans[n-1] = {n-2:1e9}
milestone_lagtime[n-1] = 0.0

temperature = 298 # K
mass = 39.9*10.0*1.66e-27 # kg
viscosity = 8.94e-14 # kg / (A * s^2)
k = k_factor * -2.5e24 / (4*avos) #2.5e24 / avos # kg * A^2/ s^2
print "k:", k
#print "velocity standard deviation:", sqrt(kb*temperature/mass) # standard deviation of a velocity distribution
#hydro_rad = 100.0 # A # defined by argparse
print "hydro_rad:", hydro_rad
friction_coeff = 6*math.pi*viscosity*hydro_rad # the friction coefficient for this particle
D = kb*temperature/friction_coeff # found the diffusion coefficient
print "D:",D, "A^2/s or ", D*1e-16, "cm^2/s"
force_func = cos1well_force
energy_func = cos1well

print "Computing the permeability according to Smol. description..."
prob_area = trap_rule(cos1well, 0.0, b, h*0.1, T=temperature, kb=kb, k=k)
prob_inv_area = trap_rule(cos1well, 0.0, b, h*0.1, T=temperature, kb=kb, k=-k)
print "prob_area:", prob_area, "A"
print "prob_inv_area:", prob_inv_area, "A"

print "permeability:", D/prob_area, "A/s"
time_estimate = estimate_time(energy_func, 0.0, b, h*0.1, T=temperature, kb=kb, k=k) / D
print "time estimate:", time_estimate

time_estimate_upside_down = estimate_time(energy_func, 0.0, b, h*0.1, T=temperature, kb=kb, k=-k) / D
print "time estimate (upside-down):", time_estimate_upside_down
#print "permeability from time estimate:", prob_inv_area / (2*time_estimate) # WRONG!!! SHOULD BE prob_area
print "permeability from time estimate:", prob_inv_area / (2*time_estimate)

# problems in the damped one might be coming from the fact that we need to only count the systems that exit to one side...
crossed_positive = 0
crossed_negative = 0
crossed_rev_positive = 0
crossed = 0
total_time = 0.0
total_time_fwd = 0.0

print "#"*30 + "\n  NOT MILESTONING\n" + "#"*30
print "now running %d simulations" % (total_num*1)
for simnum in range(total_num*1):
  milestone_pos = 0.0 # start at position 0
  #print "starting simulation:", simnum, "milestone_id:", milestone_id
  # first run the reversal phase
  
  positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, verbosity = 0, random = 1, minx = a, maxx = b, domain_a=a, domain_b=b, phase='reverse')
  if lastpos >= b: # only count these statistics from the forward phase
    #num_crossed += 1
    crossed_rev_positive += 1
    #total_time += runtime
    
  # then run the forward phase
  if reject == False:
    startvel = -velocities[0] # the starting velocity reversed
    milestone_pos = 0.0 # start at position 0
    positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, vel=startvel, verbosity = 0, random = 1, minx = a, maxx = b, domain_a=a, domain_b=b, phase='forward')
    
    if lastpos >= b: # only count these statistics from the forward phase
      #num_crossed += 1
      crossed_positive += 1
      #total_time_fwd += runtime
    elif lastpos <= a: 
      crossed_negative += 1
  
    
  # Insert another simulation here with reflecting boundary conditions...
  milestone_pos = 0.0 # start at position 0
  #print "starting simulation:", simnum, "milestone_id:", milestone_id
  # first run the reversal phase
  positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, verbosity = 0, random = 1, minx = -0.0, maxx = b, domain_a=0.0, domain_b=b, phase='reverse', reflecting = True)
  if lastpos >= b: # only count these statistics from the forward phase
    #num_crossed += 1
    #crossed_rev_positive += 1
    total_time += runtime
    
  # then run the forward phase
  if reject == False:
    startvel = -velocities[0] # the starting velocity reversed
    milestone_pos = 0.0 # start at position 0
    positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, vel=startvel, verbosity = 0, random = 1, minx = -0.0, maxx = b, domain_a=0.0, domain_b=b, phase='forward', reflecting = True)
    
    if lastpos >= b: # only count these statistics from the forward phase
      #num_crossed += 1
      crossed += 1
      total_time_fwd += runtime
    #elif lastpos <= -.1: 
    #  crossed_negative += 1
    
print "crossed_positive:", crossed_positive
print "crossed_negative:", crossed_negative
print "crossed_rev_positive:", crossed_rev_positive
if crossed_positive + crossed_negative > 0:
  beta = float(crossed_positive) / (crossed_positive + crossed_negative)
else:
  beta = 0.0

#if crossed_rev_positive > 0:
#  mfpt = total_time / crossed_rev_positive
#else:
#  mfpt = 0.0
  
if crossed > 0:
  print "crossed:", crossed
  print "total_time_fwd:", total_time_fwd
  mfpt_fwd = total_time_fwd / crossed
else:
  mfpt_fwd = 0.0
  
print "beta:", beta
#print "mean first passage time:", mfpt
print "mean first passage time forward:", mfpt_fwd

#if mfpt > 0 and mfpt_fwd > 0:
if mfpt_fwd > 0 and crossed > 1:
  #damped_perm = prob_inv_area / (2*mfpt)
  #print "damped_perm_rev:", damped_perm

  #damped_perm_fwd = prob_inv_area / (2*mfpt_fwd) # wrong
  damped_perm_fwd = prob_inv_area / (2*mfpt_fwd)
  print "damped_perm_fwd:", damped_perm_fwd, " +/- ", damped_perm_fwd / sqrt(crossed-1)

nondamped_perm = beta*D / ((1-beta)*h)
if crossed_positive > 1:
  print "nondamped_perm:", nondamped_perm, " +/- ", nondamped_perm / sqrt(crossed_positive - 1)

print "Time to complete non-milestoning portion:", time.time() - starttime, "\n"
starttime = time.time()


print "#"*30 + "\n  MILESTONING\n" + "#"*30

print "milestones:", milestones
print "n:", n
print "a:", a
print "b:", b

# run each of the milestones
for milestone_id in range(1,n-1):
  milestone_pos = milestones[milestone_id]
  print "running simulations for milestone:", milestone_id
  num_crossed = 0
  for simnum in range(total_num):
    #print "starting simulation:", simnum, "milestone_id:", milestone_id
    # first run the reversal phase
    positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, verbosity = 0, random = 1, minx = milestones[milestone_id-1], maxx = milestones[milestone_id+1], domain_a=a, domain_b=b, phase='reverse')
    #print "minx:", milestones[milestone_id-1], "maxx:", milestones[milestone_id+1], "milestone_pos:", milestone_pos
    
    # then run the forward phase
    if reject == False:
      startvel = -velocities[0] # the starting velocity reversed
      positions, velocities, forces, noises, KE, PE, energies, lastpos, runtime, reject = run_simulation(k = k, mass = mass, r = hydro_rad, force_func = force_func, energy_func = energy_func, T=temperature, viscosity=viscosity, length = length, dt = dt, pos = milestone_pos, vel=startvel, verbosity = 0, random = 1, minx = milestones[milestone_id-1], maxx = milestones[milestone_id+1], domain_a=a, domain_b=b, phase='forward')
    
      if lastpos > milestones[milestone_id+1]: # only count these statistics from the forward phase
        num_crossed += 1
        milestone_trans[milestone_id][milestone_id+1] += 1
      elif lastpos < milestones[milestone_id-1]: 
        num_crossed += 1
        milestone_trans[milestone_id][milestone_id-1] += 1
    
      milestone_lagtime[milestone_id] += runtime
  #print "milestone_trans:", milestone_trans
  if num_crossed > 0:
    milestone_lagtime[milestone_id] /= num_crossed

print "max(pos):", np.max(positions), "min(pos):", np.min(positions)
print "max(vel):", np.max(velocities), "min(vel):", np.min(velocities)
print "max(PE):", np.max(PE), "min(PE):", np.min(PE)
print "max(KE):", np.max(KE), "min(KE):", np.min(KE)

print "milestone_trans:", milestone_trans
print "milestone_lagtime:", milestone_lagtime

print "Time to complete milestoning portion:", time.time() - starttime, "\n"
starttime = time.time()

# format the dictionary to be in analyze.py format
new_trans = {}
count_trans = {}
new_lagtime = {}
for key in milestone_trans.keys():
  dest = milestone_trans[key]
  new_trans['0_%d' % key] = {} # make a new entry that contains the correct string formatting
  count_trans['0_%d' % key] = {}
  total_count = 0.0
  for destkey in dest.keys():
    total_count += milestone_trans[key][destkey]
  for destkey in dest.keys():
    new_trans['0_%d' % key]['0_%d' % destkey] = milestone_trans[key][destkey]/total_count
    count_trans['0_%d' % key]['0_%d' % destkey] = milestone_trans[key][destkey]
  new_lagtime['0_%d' % key] = milestone_lagtime[key]
    

print "new_trans:", new_trans

t_mat, index_dict = analyze.trans_dict_to_matrix(new_trans) # use functions from analyze.py to perform milestoning calcs

count_mat, dummy = analyze.trans_dict_to_matrix(count_trans) # get a count matrix
print "count_mat:", count_mat

print "index_dict:", index_dict

avg_t = analyze.avg_t_vector(new_lagtime, index_dict)

print "t_mat:", t_mat
print "avg_t:", avg_t

t_mat_inf = np.matrix(t_mat) ** 2000000
n,m = t_mat_inf.shape
q = np.zeros((n,1))
p = np.zeros((n,1))

q[0,0] = 0.5
q[1,0] = 0.5
#print "t_mat_inf:"
#pprint(t_mat_inf[:5,:5])
# calculate q stationary
q_inf = np.dot(t_mat_inf, q)

print "q_inf:", q_inf

# now calculate p stationary
print "p:"
total_p = 0.0
area_under_p = 0.0
for i in range(n):
  #print q_inf[i,0]
  p[i,0] = q_inf[i,0] * avg_t[i,0]
  total_p += p[i,0]

p_ref = float(p[1,0])

for i in range(n):
  #p[i,0] /= total_p
  p[i,0] /= p_ref
  #area_under_p += p[i,0]
  print p[i,0]
  
area_under_p = 0.0 # warning: unnormalized
for i in range(n-1):
  area_under_p += p[i,0] + p[i+1,0] # find the population  
    
area_under_p = area_under_p*h/2.0

t_mat2 = copy.deepcopy(t_mat)

q[0,0] = 0.0
q[1,0] = 1.0
t_mat[n-2][n-1] = 0.0
t_mat[0][1] = 0.0
t_mat[2][1] = 1.0
#count_mat[n-2][n-1] = 0.0
#count_mat[0][1] = 0.0
#count_mat[2][1] = 1.0

avg_t[0] = h**2 / (2*D)
avg_t[-1] = 0.0 # the last state must have an incubation time of zero
print "t_mat for mfpt:", t_mat

ident = np.matrix(np.identity(n))
mfpt = q.T * (np.linalg.inv(ident - t_mat.T) * avg_t)
mfpts = []
betas = []
nondamped_perms = []
damped_perms = []

q_mats = monte_carlo_milestoning_nonreversible_error(count_mat, avg_t, num = 1000, skip = 100)
for Qmc in q_mats:
  Tmc, avg_t_mc = rate_mat_to_prob_mat(Qmc)
  Tmc[n-2,n-1] = 1.0
  Tmc[n-1,n-1] = 0.0
  #print "Tmc:", Tmc
  Tmc_inf = np.matrix(Tmc) ** 2000000
  q = np.zeros((n,1))
  p = np.zeros((n,1))
  q[0,0] = 0.5
  q[1,0] = 0.5
  q_inf = np.dot(Tmc_inf, q)
  #print "p:"
  for i in range(n):
    p[i,0] = np.dot(q_inf[i,0], avg_t_mc[i,0])
    #print p[i,0]
  
  p_ref = float(p[1,0])
  
  for i in range(n):
    p[i,0] /= p_ref
    
  area_under_p_mc = 0.0 # warning: unnormalized
  for i in range(n-1):
    area_under_p_mc += p[i,0] + p[i+1,0] # find the population  
  area_under_p_mc = area_under_p_mc*h/2.0
  #print "area_under_p_mc:", area_under_p_mc
  
  Tmc[n-2,n-1] = 0.0
  Tmc[n-1,n-1] = 0.0
  Tmc[0,1] = 0.0
  Tmc[2,1] = 1.0
  #print "Tmc:", Tmc
  #print "avg_t_mc", avg_t_mc
  mfpt = q.T * (np.linalg.inv(ident - Tmc.T) * avg_t_mc)
        #beta_err += perr[i,0]
  mfpts.append(mfpt)
  
  
  Tmc, avg_t_mc = rate_mat_to_prob_mat(Qmc)
  Tmc[0,0] = 1.0
  Tmc[1,0] = 0.0
  Tmc[n-2,n-1] = 0.0
  Tmc[n-1,n-1] = 1.0
  Tmc_inf = np.matrix(Tmc) ** 2000000
  q[0,0] = 0.0
  q[1,0] = 1.0
  q_inf = np.dot(Tmc_inf, q)
  beta = q_inf[n-1,0]
  betas.append(beta)
  nondamped_perm = beta*D / ((1-beta)*h)
  nondamped_perms.append(nondamped_perm)
  
  damped_perm = area_under_p_mc / (2*mfpt)
  damped_perms.append(damped_perm)
  
  

print "mfpt:", mfpt, '+/-', np.std(mfpts)
print "avg of mfpt distribution:", np.mean(mfpts)

q[0,0] = 0.0
q[1,0] = 1.0
t_mat2[0][0] = 1.0
t_mat2[1][0] = 0.0
t_mat2[n-2][n-1] = 0.0
t_mat2[n-1][n-1] = 1.0
print "t_mat for q_last:", t_mat2
t_mat_inf = np.matrix(t_mat2) ** 2000000
q_inf = np.dot(t_mat_inf, q)

print "q last:", q_inf

beta = q_inf[n-1][0]


print "area_under_p:", area_under_p
damped_perm = area_under_p / (2*mfpt)
print "beta:", beta, "+/-", np.std(betas)
print "damped_perm:", damped_perm, "+/-", np.std(damped_perms) #, "BUT THIS IS WRONG!! (no its not)"
print "avg of damped_perms dist.", np.mean(damped_perms)

nondamped_perm = beta*D / ((1-beta)*h)
print "nondamped_perm:", nondamped_perm, "+/-", np.std(nondamped_perms)
print "avg of nondamped_perms dist.", np.mean(nondamped_perms)

print "permeability:", D/prob_area, "A/s"

#prob_crossing = float(num_crossed)/float(total_num)
#print "prob of crossing:", prob_crossing, "+/-", prob_crossing / sqrt(num_crossed-1)
#mfpt = total_time / total_num
#print "MFPT:", mfpt, "+/-", mfpt / sqrt(num_crossed-1)

print "Complete. Time: %0.2f" % (time.time()-starttime)



'''
func1 = np.zeros(n); func2 = np.zeros(n); func3 = np.zeros(n); func4 = np.zeros(n); xvals = np.zeros(n)
for i in range(n):
  x = a + i*h
  xvals[i] = x
  func1[i] = cos1well(x)
  func2[i] = cos2well(x)
  func3[i] = cos1barrier(x)
  func4[i] = cos2barrier(x)
'''

xmin = 0 #0.0
xmax = N #1.0
ymin = minx #np.min(W_s)
ymax = maxx
fig1 = plt.figure()
ax = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
line, = ax.plot([],[],lw=2)

xvals = np.arange(N)

plt.xlabel('x')
plt.ylabel('position')
plt.title("solution")
#plt.plot(xvals, func1, 'r', xvals, func2, 'b', xvals, func3, 'g', xvals, func4, 'y')
#plt.plot(xvals, positions, 'r') #, xvals, velocities*1e-15, 'b')
#line_ani = animation.FuncAnimation(fig1, animate, frames=num_iter, interval = 500.0, blit=True, init_func=init)
#plt.show()
