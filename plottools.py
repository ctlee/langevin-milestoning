"""
@description: A bit of a personal toolkit for plotting various things.
@author Christopher T. Lee (ctlee@ucsd.edu)
"""
import math
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib

# Initialize matplotlib settings
matplotlib.rc('font', **{'size':14})

def chunk(x, y, chunksize):
    """
    Given some x, y data break up the data point into chunks and return max, min, mean of each chunk.
    @x:         x series data
    @y:         corresponding y data
    @chunksize: size of the chunks to break into
    RETURNS
        min_env: array of min y per chunk
        max_env: array of max y per chunk
        xcenters: mean of x per chunk
        ycenters: mean of y per chunk
    """
    numchunks = y.size // chunksize # floor division
    ychunks = y[:chunksize*numchunks].reshape((-1, chunksize))
    xchunks = x[:chunksize*numchunks].reshape((-1, chunksize))
    max_env = ychunks.max(axis=1)
    min_env = ychunks.min(axis=1)
    ycenters = ychunks.mean(axis=1)
    xcenters = xchunks.mean(axis=1)
    return min_env, max_env, xcenters, ycenters

def confidenceInterval(data, confidence=0.95):
    """
    Take vertically stacked data and return the confidence interval envelope
    """
    N = data.shape[1]   # assumes as number of trials
    mean = np.mean(data, axis=0)
    stdev = sp.stats.sem(data, axis=0)
    h = stdev * sp.stats.t.ppf((1+confidence)/2, N-1)
    return mean, mean-h, mean+h
