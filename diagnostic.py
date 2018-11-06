# Various function for plotting results and for calculating error measures

import numpy as np

def lpErrorNorm(phi, phiExact,p):
    "Calculates the lp error norm of phi in comparison to"
    "phiExact"

    # calculate the error and the lp error norm
    phiError = np.abs(phi - phiExact)
    lp = np.power(sum(phiError**p)/sum(np.abs(phiExact)**p),1/p)

    return lp


def lInfErrorNorm(phi, phiExact):
    "Calculates the linf error norm (maximum normalised error) in comparison"
    "to phiExact"
    phiError = np.abs(phi - phiExact)
    return np.max(phiError)/np.max(np.abs(phiExact))
