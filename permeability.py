"""
@description: Performs a 1D Langevin dynamics simulation to validate the results
        of the milestoning permeability derivation
@authors:   Christopher T. Lee (ctlee@ucsd.edu)
            Lane Votapka (lvotapka@ucsd.edu)
@copyright Amaro Lab 2015. All rights reserved.
"""
import argparse, copy, logging, matplotlib, os, sys, pprint, pdb
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from math import log, pi, sqrt
from scipy import sparse
from collections import defaultdict
import scikits.bootstrap as skbootstrap

# Import custom modules
import plottools, traj_tools, samplefunctions, membranesystem
import membranesystem as ms
from samplefunctions import PMF, Viscosity
from markov import resample
from bootstrap import getRho
#########################################
# BRUTE FORCE CODE HERE                 #
#########################################
"""
A worker job to run the brute force calculation.
Prototype:
    arg[0]: (MembraneSystem) system - system of interest
    arg[1]: (float) length - total simulation length
    arg[2]: (float) dt - timestep
    arg[3]: (float) bq - distance between b, q
    arg[4]: (bool) cOpt - use c integrator?
"""
def bruteWorkerCrossing(args):
    # NOTE: these are not type checked
    system  = args[0]
    length  = args[1]
    dt      = args[2]
    bq      = args[3]
    cOpt    = args[4]

    if cOpt:
        milestonefunc = system.milestoneC
    else:
        milestonefunc = system.milestone
    # bookkeeping vars
    crossing = 0    # Direction of rhh crossing
    logging.debug('Calculating traversal probabilities')
    # Calculate traversal probablities
    m0, m1, accept, v0, time = milestonefunc(length = length,
            dt = dt,
            pos = -system.dz,
            vel = None,
            minx = -system.dz-bq,
            maxx = system.dz,
            phase = 'reverse',
            reflecting = False)
    if accept:
        m0, m1, accept, _, time = milestonefunc(length = length,
                dt = dt,
                pos = -system.dz,
                vel = -v0,
                minx = -system.dz-bq,  # go one further back...
                maxx = system.dz,
                phase = 'forward',
                reflecting = False)
        if accept:  # only count if it hits another milestone
            if m1 == system.dz:
                crossing = 1  # Cross positive
            elif m1 == -system.dz-bq:
                crossing = -1 # Cross negative
    logging.info('accept %d\n'%(accept))
    return crossing, -1  # tuple (crossing, time)

"""
A worker job to run the brute force calculation.
Prototype:
    arg[0]: (MembraneSystem) system - system of interest
    arg[1]: (float) length - total simulation length
    arg[2]: (float) dt - timestep
    arg[3]: (float) bq - distance between b, q
    arg[4]: (bool) cOpt - use c integrator?
"""
def bruteWorkerTimes(args):
    # NOTE: these are not type checked
    system  = args[0]
    length  = args[1]
    dt      = args[2]
    bq      = args[3]
    cOpt    = args[4]

    if cOpt:
        milestonefunc = system.milestoneC
    else:
        milestonefunc = system.milestone
    # bookkeeping vars
    fwdtime = -1    # Time to go forward
   
    logging.debug('Calculating forward times')
    # Calculate forward times   
    # 2/24/16 Relaxing the conditions for the trajectory 
    # to allow accept over large barriers
    accept = False
    while not accept:
        m0, m1, accept, v0, time = milestonefunc(length = length,
                dt = dt,
                pos = -system.dz,
                vel = None,
                minx = -system.dz-bq,
                maxx = system.dz,
                phase = 'reverse',
                reflecting = False)  
    logging.debug('Starting forward phase')
    m0, m1, accept, _, time = milestonefunc(length = length,
        dt = dt,
        pos = -system.dz,
        vel = -v0,
        minx = -system.dz,
        maxx = system.dz,
        phase = 'forward',
        reflecting = True)
    logging.info('accept %d, time %e\n'%(accept, time))
    if accept:  # Count if it hits another milestone
        if m1 == system.dz:
            fwdtime = time
    return 0, fwdtime  # tuple (crossing, time)

"""
Helper function to check if the files exist already. If not fill first line with
metadata.
@args system: the system to write about
"""
def checkBruteFD(system):
    if not os.path.exists('datasets/' + system.name + '_crossing'):
        try:
            with open('datasets/' + system.name + 
                    '_crossing', 'wb') as crossingfd:
                crossingfd.write('#neg pos length dt bq #name: cOpt\n')
        except IOError as e:
            logging.error('I/O error(%s: %s)'%(e.errno, e.strerror))
    if not os.path.exists('datasets/' + system.name + '_times'):
        try:
            with open('datasets/' + system.name + '_times', 'wb') as timefd:
                timefd.write('#time length dt bq #name: cOpt\n')
        except IOError:
            logging.error('I/O error(%s: %s)'%(e.errno, e.strerror))

"""
Wrapper around the worker function
@args See bruteWorker prototype
@args out_q: (Queue) to write results to
"""
def bruteMPWorkerCrossing(args):
    out_q = args[5]
    out_q.put(bruteWorkerCrossing(args[0:5]))


"""
Wrapper around the worker function
@args See bruteWorker prototype
@args out_q: (Queue) to write results to
"""
def bruteMPWorkerTimes(args):
    out_q = args[5]
    out_q.put(bruteWorkerTimes(args[0:5]))

"""
Writer function for multiprocessing
"""
def bruteMPlog(system, length, dt, bq, cOpt, q):
    checkBruteFD(system)
    with open('datasets/' + system.name + '_crossing', 'ab') as crossingfd, \
            open('datasets/' + system.name + '_times', 'ab') as timefd:
        while 1:
            m = q.get()
            if m == 'kill':
                logging.debug('bruteMPlog listener killed')
                break
            else:
                if m[0] != 0: 
                    if m[0] == -1:
                        crossingfd.write('1 0 %1.2e %1.2e %d #%s: cOpt %i\n'
                                %(length, dt, bq, system.name, cOpt))
                    elif m[0] == 1:
                        crossingfd.write('0 1 %1.2e %1.2e %d #%s: cOpt %i\n'
                                %(length, dt, bq, system.name, cOpt))
                    else:
                        logging.error("Job output is incorrect")
                    crossingfd.flush()
                if m[1] != -1:
                    timefd.write('%1.10e %1.2e %1.2e %d #%s: cOpt %i\n'
                            %(m[1], length, dt, bq, system.name, cOpt))
                    timefd.flush()

def bruteMPCrossing(system, length = 1, dt = 2e-15, bq = 1,
        cOpt = True, numSims = 10000):
    logging.debug("## " + "Brute Force Calculation MP" + " ##")
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool()
    watcher = pool.apply_async(bruteMPlog, 
            (system, length, dt, bq, cOpt, out_q))
    jobs = []
    for i in np.arange(0, numSims, 1):
        jobs.append((system, length, dt, bq, cOpt, out_q))
    logging.info('Starting %d jobs'%(len(jobs)))
    pool.map(bruteMPWorkerCrossing, iter(jobs))

    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()

def bruteMPTimes(system, length = 1, dt = 2e-15, bq = 1,
        cOpt = True, numSims = 10000):
    logging.debug("## " + "Brute Force Calculation MP" + " ##")
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool()
    watcher = pool.apply_async(bruteMPlog, 
            (system, length, dt, bq, cOpt, out_q))
    jobs = []
    for i in np.arange(0, numSims, 1):
        jobs.append((system, length, dt, bq, cOpt, out_q))
    logging.info('Starting %d jobs'%(len(jobs)))
    pool.map(bruteMPWorkerTimes, iter(jobs))

    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()


def bruteForce(system, length = 1, dt = 2e-15, bq = 1, 
        cOpt = True, numSims = 100000):
    logging.debug("## " + "Brute Force Calculation" + " ##")
    
    checkBruteFD(system)
    with open('datasets/' + system.name + '_crossing', 'ab') as crossingfd, \
            open('datasets/' + system.name + '_times', 'ab') as timefd:
        for sim in np.arange(0, numSims, 1):
            logging.debug("Starting simulation %d"%sim)
            crossing, _ = bruteWorkerCrossing((system, length, dt, bq, cOpt))
            _, time = bruteWorkerTimes((system, length, dt, bq, cOpt))
            if crossing == -1:
                crossingfd.write('1 0 %1.2e %1.2e %d #%s: cOpt %i\n'
                        %(length, dt, bq, system.name, cOpt))
            elif crossing == 1:
                crossingfd.write('0 1 %1.2e %1.2e %d #%s: cOpt %i\n'
                        %(length, dt, bq, system.name, cOpt))
            if time > 0:
                timefd.write('%1.10e %1.2e %1.2e %d #%s: cOpt %i\n'
                        %(time, length, dt, bq, system.name, cOpt))
    return 0

def processBrute(system, prefix='datasets/'):
    print("-----Process Brute------")
    try: 
        crossNeg, crossPos, length, dt, bq = np.loadtxt(prefix + system.name +
                '_crossing', comments='#', skiprows=1, unpack=True)
    except ValueError as e:
        logging.error('No crossing statistics, cannot calculate permeability.')
        return 1
    timingStats = []
    length2 = []
    dt2 = []
    try:
        timingStats, length2, dt2, _ = np.loadtxt(prefix + system.name + 
                '_times', comments='#', skiprows=1, unpack=True)
    except ValueError as e:
        logging.warn('No timing statistics.')
    
    length = np.concatenate((length, length2))
    dt = np.concatenate((dt, dt2))
    
    if not np.all(np.equal(length,length[0])):
        logging.warning('Mismatch in simulation lengths.')

    if not np.all(np.equal(dt, dt[0])): 
        logging.warning('Mismatch in timestep.')

    if not np.all(np.equal(bq, bq[0])):
        logging.error('BQ is inconsistent among runs.')
        return 1
    else:
        bq = bq[0]

    crossNeg = np.sum(crossNeg)
    crossPos = np.sum(crossPos)
    logging.info('Brute crossPos: %d; crossNeg: %d'%(crossPos, crossNeg))
    
    rho = float(crossPos)/float(crossPos + crossNeg)

    # Bootstrap Rho to get confidence in probability
    transitions = np.concatenate([np.ones(crossPos), np.ones(crossNeg)*-1], 
            axis=0)
    if crossPos and crossNeg:
        rhoCI = skbootstrap.ci(data=transitions, statfunction=getRho,
                output='errorbar', n_samples=10000, method='pi')
    else:
        rhoCI = np.zeros(2)
    logging.debug("D = %f"%(system.getD(-system.dz)))
    logging.debug("Brute bq = %f"%(bq))
    print("Brute Rho (n=%d): %e; "%(crossPos+crossNeg, rho) +
            '95%% CI %e'%((rhoCI[0]+rhoCI[1])/2))
    if len(timingStats) != 0:
        mfpt = np.mean(timingStats)
        CI = skbootstrap.ci(data=timingStats, statfunction=np.mean, 
                output='errorbar', n_samples=10000, method='bca')
        stdev = sqrt(np.sum(np.power(timingStats-mfpt,2))/len(timingStats))
        print('Brute MFPT (n=%d): %e +/- %e, '%(timingStats.size, mfpt, stdev) + 
                '95%% CI low: %e, high: %e s'%(CI[0], CI[1]))
        Pdamped = system.cumulativeProbDist()/(2*mfpt) * 1e-8 # A/s -> cm/s
        Phigh = system.cumulativeProbDist()/(2*(mfpt+CI[1])) * 1e-8 # A/s -> cm/s
        Plow = system.cumulativeProbDist()/(2*(mfpt-CI[0])) * 1e-8
        diff = ((Phigh-Pdamped)+ (Plow-Pdamped))/2
        logdiff = ((log(Pdamped,10)-log(Phigh,10)) + (log(Plow,10)-log(Pdamped,10)))/2
        print("Brute MFPT-ISD: %e +/- %e cm/s; %f +/- %f"
                %(Pdamped, diff, log(Pdamped, 10), logdiff))
        # Plot a histogram of the MFPT 
        fig = plt.figure(99, facecolor='white', figsize=(7,5.6))
        ax1 = fig.add_subplot(111)
        count, bins, ignored = ax1.hist(timingStats*1e6, 100)
        ax1.errorbar(mfpt*1e6, np.amax(count)/2, xerr=CI*1e6, 
                fmt='.', ecolor='r', color='r', elinewidth=1, capsize=2)
        ax1.set_ylabel(r'Probability [au]')
        ax1.set_xlabel(r'First PassageTime [$\mu s$]')
        ax1.margins(0,0.05)
        fig.savefig('figures/%s_brutemfpt.png'%(system.name), dpi=300)
        plt.close('all')

    if rho > 0:
        P = rho*system.getD(-system.dz) / ((1-rho) * bq) * 1e-8  # A/s -> cm/s
        rhigh = rho+rhoCI[1]
        Phigh = rhigh*system.getD(-system.dz) / ((1-rhigh)*bq) * 1e-8
    
        rlow = rho-rhoCI[0]
        Plow = rlow*system.getD(-system.dz) / ((1-rlow)*bq) * 1e-8
        diff = Phigh-P
        print("Brute PBCP: %e +/- %e cm/s; %f +/- %f"%(P, diff, 
            log(P,10), (log(Phigh,10)-log(P,10) + log(P,10)-log(Plow,10))/2))
    else:
        logging.error('Invalid value of rho.')
    return 0

#########################################
#  MILESTONING CODE HERE                #
#########################################
def milestoneWorker(args):
    # NOTE: these are not type checked
    system      = args[0]
    milestones  = args[1]
    index       = args[2]
    length      = args[3]
    dt          = args[4]
    cOpt        = args[5]
    if cOpt:
        milestonefunc = system.milestoneC
    else:
        milestonefunc = system.milestone
    crossing = 0
    crossingTime = -1
   
    pos = milestones[index]
    # Prevent going OOB by enforcing equal milestone spacing at termini
    if index == 0:
        minx = milestones[index] - (milestones[index+1] - milestones[index])
    else:
        minx = milestones[index-1]

    if index == len(milestones)-1:
        maxx = milestones[index] + (milestones[index] - milestones[index-1])
    else:
        maxx = milestones[index+1]
    
    m0, m1, accept, v0, time = system.milestone(length = length,
            dt = dt,
            pos = pos,
            vel = None,
            minx = minx,
            maxx = maxx,
            phase = 'reverse',
            reflecting = False)
    if accept:
        m0, m1, accept, _, time = system.milestone(length = length,
                dt = dt,
                pos = pos,
                vel = -v0,
                minx = minx,
                maxx = maxx,
                phase = 'forward',
                reflecting = False)
        if accept:  # only count if it hits another milestone
            # remove boundary values
            if m1 == maxx:
                if not index == len(milestones)-1:
                    crossing = 1 
            elif m1 == minx:
                if not index == 0:
                    crossing = -1 
            crossingTime = time
    logging.info("index %0.1f, crossing %d, crossingTime, %f\n"%(index, 
            crossing, crossingTime))
    return index, crossing, crossingTime

"""
Helper function to check if the files exist already. If not fill first line with
metadata.
@args system: the system to write about
"""
def checkMilestoneFD(system, milestones):
    if not os.path.exists('datasets/' + system.name + '_milestonestats'):
        try:
            with open('datasets/' + system.name + '_milestonestats', 'wb') as timefd:
                timefd.write('#start end time length dt #name: cOpt\n')
        except IOError:
            logging.error('I/O error(%s: %s)'%(e.errno, e.strerror))

def checkMilestones(system, milestones):
    if os.path.exists('datasets/' + system.name + '_milestones'):
        # Check if the milestones are the same 
        savedMilestones = np.load('datasets/' + system.name + '_milestones')
        # TODO: use better array comparison
        if not np.all(np.equal(milestones, savedMilestones)):
            logging.error('Milestones do not match. Exiting')
            sys.exit(1)
    else:
        try: 
            with open('datasets/' + system.name + 
                    '_milestones', 'wb') as milestonesfd:
                np.save(milestonesfd, milestones)
        except IOError as e:
            logging.error('I/O error(%s: %s)'%(e.errno, e.strerror))

"""
Wrapper around the worker function
@args See milestoneWorker prototype
@args out_q: (Queue) to push results to
"""
def milestoneMPWorker(args):
    out_q = args[6]
    out_q.put(milestoneWorker(args[0:6]))

"""
Writer function for multiprocessing
"""
def milestoneMPlog(system, milestones, length, dt, cOpt, q):
    checkMilestoneFD(system, milestones)
    with open('datasets/' + system.name + '_milestonestats', 'ab') as fd:
        while 1:
            m = q.get()
            if m == 'kill':
                logging.debug('milestoneMPlog listener killed')
                break
            else:
                if m[1] != 0:
                    # start end time length dt #name: cOpt
                    fd.write('%0.2f %0.2f %1.10e %1.2e %1.2e #%s: cOpt %i\n'
                            %(m[0], m[0] + m[1], m[2], 
                            length, dt, system.name, cOpt))
                    fd.flush()

def milestoneMP(system, milestones, length = 1, dt = 2e-15, 
        cOpt = True, numSims = 10000, focus = None):
    logging.debug("## " + "Running Milestoning MP" + " ##")
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool()
    watcher = pool.apply_async(milestoneMPlog, 
            (system, milestones, length, dt, cOpt, out_q))
    jobs = []
    checkMilestones(system, milestones)
    if focus is not None:
        for index in focus:  # run the middle milestones
            for sim in np.arange(0, numSims, 1):
                jobs.append((system, milestones, index, length, dt, cOpt, out_q))
    else:
        for index in xrange(0, len(milestones)):  # run the middle milestones
            for sim in np.arange(0, numSims, 1):
                jobs.append((system, milestones, index, length, dt, cOpt, out_q))
    logging.info('Starting %d jobs'%(len(jobs)))
    pool.map(milestoneMPWorker, iter(jobs))
    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()

"""
Non-functioning TODO: fix it
"""
def milestoning(system, milestones, length = 1, dt = 2e-15,
        cOpt = True, numSims = 10000):
    logging.debug("## " + "Running Milestoning Serial" + " ##")
    checkMilestones(system, milestones)    
    checkMilestoneFD(system, milestones)
    with open('datasets/' + system.name + '_milestonestats', 'ab') as fd:
        for index in xrange(0, len(milestones)):  # run the middle milestones
            for sim in np.arange(0, numSims, 1):
                start, cross, time = milestoneWorker(system, milestones, index, 
                        length, dt, cOpt)
                if cross != 0:
                    # start end time length dt #name: cOpt
                    fd.write('%0.2f %0.2f %1.10e %1.2e %1.2e #%s: cOpt %i\n'
                            %(start, start + cross, time, length, dt, 
                            system.name, cOpt))
                    fd.flush()

def processMilestones(system, milestones, prefix='datasets/'):
    """
    TODO: Update to use sparse matrices.
    """
    print("-----Process Milestone------")
    checkMilestones(system, milestones)
    try:
        stats = np.loadtxt(prefix + system.name +
                '_milestonestats', comments='#', skiprows=1)
    except IOError as e:
        logging.error('I/O Error(%d): %s'%(e.errno, e.strerror))
        return 1
    if stats.size == 0:
        logging.error('No milestoning statistics.')
        return 1
    
    length = stats[:,3]
    dt = stats[:,4]
    if not  np.all(np.equal(length,length[0])):
        logging.warning('Mismatch in simulation lengths.')
    if not np.all(np.equal(dt, dt[0])): 
        logging.warning('Mismatch in timestep.')

    logging.info("Milestones:\n" + str(milestones))
    N = len(milestones)
    transCount = np.matrix(np.zeros((N, N)))
    
    transTimesDict = defaultdict(np.array)
    lifetimesDict = defaultdict(np.array)
    for stat in stats:
        key = (stat[0], stat[1])
        if stat[0] == stat[1]:
            print stat
        transCount[stat[0], stat[1]] += 1
        if transTimesDict.has_key(key):
            transTimesDict[key] = np.append(transTimesDict[key], stat[2])     
        else:
            transTimesDict[key] = np.array(stat[2])
        key = stat[0]
        if lifetimesDict.has_key(key):
            lifetimesDict[key] = np.append(lifetimesDict[key], stat[2])
        else:
            lifetimesDict[key] = np.array(stat[2])
    #logging.info("Unnormalized transition count:")
    #logging.info(transCount)
    rowSum = np.matrix.sum(transCount, axis=1)
    zeroIndices = rowSum == 0
    rowSum[zeroIndices] = np.inf
    K = np.divide(transCount,rowSum)
    
    logging.info("Transition kernel (K):\n" + str(K))
    logging.info("Count per milestone:\n" + str(rowSum.T))
    logging.info("transCount:\n" + str(transCount))
    
    K = transCount/np.matrix.sum(transCount, axis=1)

    logging.info("K:\n" + str(K)) 
    #K = np.nan_to_num(0)    # Replace nans with 0
    logging.info("Transition kernel (K):\n" + str(K))
    logging.info("Count per milestone:\n" + 
            str(np.matrix.sum(transCount, axis=1)))

    # Calculate the lifetimes
    avgTimes = np.matrix(np.zeros((N, N)))
    for key in transTimesDict:
        avgTimes[key[0], key[1]] = np.mean(transTimesDict[key])

    lifetimes = np.zeros((N))
    for key in lifetimesDict:
        lifetimes[key] = np.mean(lifetimesDict[key])
    # convert to matrix
    lifetimes = np.matrix(lifetimes)
    logging.info("Lifetimes:\n" + str(lifetimes))
    
    I = np.identity(N)
   
    mfpts, rhos, P_MFPTs, P_PBCPs = resample(transCount, lifetimes, system, milestones)

    # Setup initial flux
    q = np.zeros(N)
    q[0] = 0.5
    q[1] = 0.5
   
    """
    # Print the transition statistics
    for row in np.arange(0,N,1):
        print row, rowSum[row]
        if row == 0:
            print K[row, row], K[row,row+1]
        elif row == N-1:
            print K[row, row-1], K[row, row]
        else:
            print K[row,row-1], K[row, row], K[row,row+1]
    """ 
    w, vl= LA.eig(K, left=True, right=False)
    if w[-1].real == 1:
        logging.info("Left eigenvalue %f"%(w[-1].real))
        qstat = vl[:,-1].real    # Get first column
        qstat = np.matrix(qstat.T)
        qstat = qstat/LA.norm(qstat)
        logging.info("Normalized Stationary Flux (qstat):\n" + str(qstat))
    else:
        logging.info("Largest eigenvalue (w[-1]): %f"%(w[-1].real))
        logging.info("Corresponding eigenvector:\n" + str(vl[:,-1].real))
        logging.info("Could not find unit left eigenvalue. " + \
                "Using power method instead.")
        # Calculation of the stationary flux by power:
        # This maybe slow for large transition kernels. Further the 
        # number of steps to take may be a lot.
        Kinf = K**99999999 # Some big number. TODO check if it's big enough
        qstat = q.dot(Kinf)
        qstat = qstat/LA.norm(qstat)
        logging.info("Stationary flux via power:\n" + str(qstat))

    # Compute the stationary probability
    tabsorb = np.copy(lifetimes)
    tabsorb[0,N-1] = 0
    pstat = np.multiply(qstat, tabsorb)

    logging.info("Stationary probability (pstat):\n" + str(pstat))

    sumProb = np.sum(pstat)
    pstat = pstat/sumProb   # Normalize to 1
    pstat = pstat/pstat[0,0] # set first point to 0

    q = np.zeros((N,1))
    q[0,0] = 1
    K[N-1] = 0
    aux = LA.solve(I-K, tabsorb.T);
    mfpt = q.T.dot(aux)
    print "Milestoning MFPT: %e +/- %e s"%(mfpt, np.std(mfpts))

    # Setup the fancy looping boundaries where 0 and N-1 are absorbing
    K[0,0] = 1  # Absorbing
    K[0,1] = 0
    K[N-1] = 0
    K[N-1, N-1] = 1 # Absorbing
    
    Kinf = K**99999999
    q = np.zeros(N)
    q[1] = 1.0
    qstat = q.dot(Kinf)
    logging.info("Looping Stationary Flux (qstat):\n" + str(qstat))
    rho = qstat[0,N-1]
    print("Milestoning Rho: %e +/- %e"%(rho, np.std(rhos)))
    
    """ 
    # It's better to integrate over the non-biased PMF
    Pdamped = np.trapz(pstat, milestones)/(2*mfpt) * 1e-8 # A/s -> cm/s
    print("Milestoning MFPT-ISD: %e +/- %e cm/s; %f +/- %f"
            %(Pdamped, np.std(dampedPs), log(Pdamped, 10), logdiff))
    """

    if rho != 0:
        Pnondamped = rho*system.getD(milestones[0])/ \
                ((1-rho)*(milestones[1] - milestones[0])) * 1e-8
        dpstd = np.std(P_PBCPs)
        Phigh = Pnondamped + dpstd
        Plow = Pnondamped - dpstd

        if Plow < 0:
            logdiff = log(Phigh,10)-log(Pnondamped,10)
        else:
            logdiff = ((log(Phigh,10)-log(Pnondamped,10)) + (log(Pnondamped,10)-log(Plow,10)))/2
        print("Milestoning PBCP: %e +/- %e cm/s; %f +/- %f"
                %(Pnondamped, dpstd, log(Pnondamped, 10), logdiff))

    
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
    q[0] = 1
    
    
    
    # Print the transition statistics
    for row in np.arange(0,N,1):
        print row, rowSum[row]
        if row == 0:
            print K[row, row], K[row,row+1]
        elif row == N-1:
            print K[row, row-1], K[row, row]
        else:
            print K[row,row-1], K[row, row], K[row,row+1]
    
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
    
    print qstat
    print lifetimes
    # Compute the stationary probability
    pstat = np.multiply(qstat, lifetimes)

    sumProb = np.sum(pstat)
    pstat = pstat/sumProb   # Normalize to 1
    pstat = pstat/pstat[0,0] # set first point to 0
    
    Pdamped = np.trapz(pstat, milestones)/(2*mfpt) * 1e-8 # A/s -> cm/s
    dpstd = np.std(P_MFPTs)
    Phigh = Pdamped + dpstd
    Plow = Pdamped - dpstd
    if Plow < 0:
        logdiff = log(Phigh,10)-log(Pdamped,10)
    else:
        logdiff = ((log(Phigh,10)-log(Pdamped,10)) + (log(Pdamped,10)-log(Plow,10)))/2
    print("Milestoning PBC MFPT-ISD: %e +/- %e cm/s; %f +/- %f"
            %(Pdamped, dpstd, log(Pdamped, 10), logdiff))
    
    kb = 0.0019872041   # kcal/mol/A
    F = -kb*system.T*np.log(pstat)
    scalefactor = 1.43929254302   # conversion to kcal/mol from kgA^2/S
    
    fig = plt.figure(1, facecolor='white', figsize=(7,5.6))
    ax1 = fig.add_subplot(111)
    pmf = system.pmf(system.z)*scalefactor
    ax1.plot(system.z, pmf, linewidth=2, color='k', 
            linestyle='-', label='Supplied')
    ax1.plot(milestones, np.asarray(F)[0], linewidth=2, linestyle='--',
            color='r', label='Milestoning')
    ax1.set_ylabel(r'PMF [$kcal/mol$]')
    ax1.set_xlabel(r'Position [$\AA$]')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, 
            loc = 'upper left',
            fontsize = 'small',
            frameon = False)
    ax1.margins(0,0.05)
    if system.name == 'flat':
        ax1.set_ylim([-0.5,0.5])
    fig.savefig('figures/%s_pmf_calc.png'%(system.name), dpi=300)
    plt.close('all')
