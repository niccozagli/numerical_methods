import numpy as np

def cosBell(x, alpha=0, beta=0.5):
    "Function defining a cosine bell as a function of position, x"
    "between alpha and beta with default parameters 0, 0.5"
    "padding the rest of the spatial interval at zero"
    width = beta - alpha
    bell = lambda x: 0.5*(1 - np.cos(2*np.pi*(x-alpha)/width))
    return np.where((x<beta) & (x>=alpha), bell(x), 0.)

def squareWave(x,alpha,beta):
    "A square wave as a function of position, x, which is 1 between alpha"
    "and beta and zero elsewhere. The initialisation is conservative so"
    "that each phi contains the correct quantity integrated over a region"
    "a distance dx/2 either side of x"

    phi = np.zeros_like(x)

    # The grid spacing (assumed uniform)
    dx = np.abs(x[1] - x[0])

    # Set phi away from the end points (assume zero at the end points)
    for j in range(1,len(x)-1):
        # edges of the grid box
        xleft = x[j] - 0.5*dx
        xright = x[j] + 0.5*dx
        
        #integral quantity of phi
        phi[j] = max((min(beta, xright) - max(alpha, xleft))/dx, 0)

    return phi
