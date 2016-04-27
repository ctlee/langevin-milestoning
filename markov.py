import numpy as np
import math, random
from math import exp, log
import scipy.linalg as LA
import membranesystem

def count_mat_to_rate_mat(count_matrix, avg_t):
  ''' converts a count matrix to a markov rate matrix (where each entry is an effective kinetic rate constant)'''
  n = np.shape(count_matrix)[0]
  rate_matrix = np.matrix(np.zeros((n,n)))
  sum_vector = np.zeros(n)
  for i in range(n): # first make the sum of all the counts
    for f in range(n):
      sum_vector[f] += count_matrix[i,f]


  for i in range(n):
    for f in range(n):
      if f == i: continue
      if sum_vector[f] == 0 or avg_t[f] == 0.0:
        rate_matrix[i,f] = 0.0
      else:
        rate_matrix[i,f] = count_matrix[i,f] / (sum_vector[f] * avg_t[f])
      rate_matrix[f,f] -= rate_matrix[i,f]

  return rate_matrix, sum_vector

def rate_mat_to_prob_mat(rate_matrix):
  ''' converts a rate matrix into probability matrix (kernel) and an incubation time vector'''
  n = rate_matrix.shape[0]
  P = np.matrix(np.zeros((n,n)))
  prob_matrix = np.matrix(np.zeros((n,n)))
  sum_vector = np.zeros(n)
  avg_t = np.zeros((n,1))
  for i in range(n): # first make the sum of all the rates
    for j in range(n):
      if j == i: continue
      sum_vector[j] += rate_matrix[i,j]


  for i in range(n):
    for j in range(n):
      if j == i: continue
      if sum_vector[j] == 0:
        prob_matrix[i,j] = 0.0
        #avg_t[j] = 0.0
      else:
        prob_matrix[i,j] = rate_matrix[i,j] / sum_vector[j]

    if sum_vector[i] != 0.0:
      avg_t[i,0] = 1.0 / sum_vector[i]
    else: # then the sum vector goes to zero, make the state go to itself
      prob_matrix[i,i] = 1.0
      #avg_t[i] = something???

  return prob_matrix, avg_t

def monte_carlo_milestoning_nonreversible_error(N, avg_t, num = 20, skip = 0):
  ''' Samples a distribution of rate matrices that are nonreversible
      using Markov Chain Monte Carlo method

      The distribution being sampled is:
      p(Q|N) = p(Q)p(N|Q)/p(N) = p(Q) PI(q_ij**N_ij * exp(-q_ij * N_i * t_i))

      N = count matrix
      avg_t = incubation times

  '''
  Q0, N_sum = count_mat_to_rate_mat(N, avg_t) # get a rate matrix and a sum of counts vector
  m = N.shape[0] # the size of the matrix
  Q = Q0
  Q_mats = []

  for counter in range(num*(skip+1)):
    Qnew =  np.matrix(np.zeros((m,m))) #np.matrix(np.copy(T))
    for i in range(m): # rows
      for j in range(m): # columns
        Qnew[i,j] = Q[i,j]

    for i in range(m): # rows
      for j in range(m): # columns
        if i == j: continue
        if Qnew[i,j] == 0.0: continue
        if Qnew[j,j] == 0.0: continue
        delta = random.expovariate(1.0/(Qnew[i,j])) - Qnew[i,j] # so that it's averaged at zero change, but has a minimum value of changing Q[j,j] down only to zero

        if np.isinf(delta): continue
        r = random.random()

        # NOTE: all this math is being done in logarithmic form first (log-likelihood)
        new_ij = N[i,j] * log(Qnew[i,j] + delta) - ((Qnew[i,j] + delta) * N_sum[j] * avg_t[j])
        old_ij = N[i,j] * log(Qnew[i,j]) - ((Qnew[i,j]) * N_sum[j] * avg_t[j])
        p_acc = (new_ij - old_ij) # + (new_jj - old_jj)
        if log(r) <= p_acc: # this can be directly compared to the log of the random variable
          Qnew[j,j] = Qnew[j,j] - delta
          Qnew[i,j] = Qnew[i,j] + delta

    if skip == 0 or counter % skip == 0: # then save this result for later analysis
      Q_mats.append(Qnew)
    Q = Qnew
  return Q_mats

def resample(transCount, lifetimes, system, milestones):
    N = transCount.shape[0] # the size of the matrix

    rhos = []
    mfpts = []
    dampedPs = []
    nondampedPs = []
    Qsamples = monte_carlo_milestoning_nonreversible_error(transCount.T, lifetimes.T, 
            num=1000, skip=100)
   
    I = np.identity(N)

    for Qs in Qsamples:
        K, lifetimes = rate_mat_to_prob_mat(Qs)
        K = K.T
        lifetimes = lifetimes.T
        
        q = np.zeros(N)
        q[0] = 0.5
        q[1] = 0.5
        
        w, vl= LA.eig(K, left=True, right=False)
        if w[-1].real == 1:
            qstat = vl[:,-1].real    # Get first column
            qstat = np.matrix(qstat.T)
            qstat = qstat/LA.norm(qstat)
        else:
            Kinf = K**99999999
            qstat = q.dot(Kinf)
            qstat = qstat/LA.norm(qstat)
      
        # Compute the stationary probability
        tabsorb = np.copy(lifetimes)
        tabsorb[0,N-1] = 0

        q = np.zeros((N,1))
        q[0] = 0.5
        q[1] = 0.5
        K[N-1] = 0
        aux = LA.solve(I-K, tabsorb.T);
        mfpt = q.T.dot(aux)
        mfpt = mfpt[0,0]
        mfpts.append(mfpt)

        # Setup the fancy looping boundaries where 0 and N-1 are absorbing
        K[0,0] = 1  # Absorbing
        K[0,1] = 0
        K[N-1] = 0
        K[N-1, N-1] = 1 # Absorbing
        
        Kinf = K**99999999
        q = np.zeros(N)
        q[1] = 1.0
        qstat = q.dot(Kinf)
        rho = qstat[0,N-1]
        rhos.append(rho)
      
        if rho != 0:
            Pnondamped = rho*system.getD(milestones[0])/ \
                    ((1-rho)*(milestones[1] - milestones[0])) * 1e-8
        nondampedPs.append(Pnondamped) 
        
        ########################## 
        # Calculate Free Energy  #
        ##########################
        # reset from previous calc
        K[0,0] = 0
        # Setup periodic boundary condition
        K[N-1] = 0
        K[N-1,N-2] = 0.5
        K[N-1,0] = 0.5
        K[0,1] = 0.5
        K[0,N-1] = 0.5
        
        # Setup initial flux
        q = np.zeros(N)
        q[0] = 0.5
        q[1] = 0.5        
        w, vl= LA.eig(K, left=True, right=False)
        if w[-1].real == 1:
            qstat = vl[:,-1].real    # Get first column
            qstat = np.matrix(qstat.T)
            qstat = qstat/LA.norm(qstat)
        else:
            # Calculation of the stationary flux by power:
            # This maybe slow for large transition kernels. Further the 
            # number of steps to take may be a lot.
            Kinf = K**99999999 # Some big number. TODO check if it's big enough
            qstat = q.dot(Kinf)
            qstat = qstat/LA.norm(qstat)
        # Compute the stationary probability
        pstat = np.multiply(qstat, lifetimes)

        sumProb = np.sum(pstat)
        pstat = pstat/sumProb   # Normalize to 1
        pstat = pstat/pstat[0,0] # set first point to 0
         
        Pdamped = np.trapz(pstat, milestones)/(2*mfpt) * 1e-8 # A/s -> cm/s
        dampedPs.append(Pdamped)
    return mfpts, rhos, dampedPs, nondampedPs
