import numpy as np

def cosBell(x, alpha=0, beta=0.5):
    "Function defining a cosine bell as a function of position, x"
    "between alpha and beta with default parameters 0, 0.5"
    "padding the rest of the spatial interval at zero"
    width = beta - alpha
    bell = lambda x: 0.5*(1 - np.cos(2*np.pi*(x-alpha)/width))
    return np.where((x<beta) & (x>=alpha), bell(x), 0.)
