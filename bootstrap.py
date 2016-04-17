import sklearn
import numpy as np
import warnings
import scikits.bootstrap as skbootstrap

def getRho(data):
    N = len(data)
    total = np.sum(data)
    crossNeg = (N-total)/2
    crossPos = N-crossNeg
    return float(crossPos)/(crossPos + crossNeg)

def bootstrapRho(data, n_samples=10000, alpha=0.05):
    N = len(data)
    rhos = []
    # Generate n_samples new means from transition data
    for i in np.arange(0, n_samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resample = sklearn.utils.resample(data)
        total = np.sum(resample)
        crossNeg = (N-total)/2
        crossPos = N-crossNeg
        rhos.append(float(crossPos)/(crossPos + crossNeg))
    return np.std(rhos, ddof = N-1)

def bootstrapSKRho(data, n_samples=10000, alpha=0.05):
    """
    This function samples the confidence interval of n_samples means
    """
    N = len(data)
    rhos = []
    # Generate n_samples new means from transition data
    for i in np.arange(0, n_samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resample = sklearn.utils.resample(data)
        total = np.sum(resample)
        crossNeg = (N-total)/2
        crossPos = N-crossNeg
        rhos.append(float(crossPos)/(crossPos + crossNeg))
    return skbootstrap.ci(data=rhos, statfunction=np.mean, output='errorbar')
