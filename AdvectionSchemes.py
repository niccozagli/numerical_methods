#In this module we will define some advection schemes to integrate the linear
#advection equation. In particular, we have defined the following schemes:
# - the upwind scheme (which is FTBS if the velocity 'u' of the fluid is u>=0
# and FTFS if u<0
# - BTCS
# - Lax Wendroff and Warming Beam
# - TVD
# Each scheme returns the evoluted variable 'phi' at the last time tmax and the whole"
# temporal evolution of the first 'M' and second 'V' moment and the total variation 'TV'
# of the solution
# Each scheme has 4 inputs : 'x' the spatial grid , 't' the temporal grid,
# 'phiold' the initial solution that will evolve in time and 'c', the Courant number

# Recall : the schemes are valid for a system with periodic boundary conditions"

import numpy as np
import scipy.linalg as linalg
from conservation_and_total_variation import *

def upwind( x , t , phiold , c ):
    "This function calls the right scheme depending on the velocity of the fluid"
    "Input and output described at the beginning of the module"
    if(c>=0):
        phi_upwind , M , V , TV = FTBS( x , t , phiold , c )
    else:
        phi_upwind , M , V , TV = FTFS(x , t , phiold , c )
    return phi_upwind , M , V , TV

def FTBS( x , t , phiold , c ):
    "FTBS scheme. Input and output described at the beginning of the module"
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]-c*(phiold[j%nx]-phiold[(j-1)%nx])

        phiold = phi.copy()
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)
    return phiold , M , V , TV

def FTFS( x , t , phiold , c ):
    "FTFS scheme. Input and output described at the beginning of the module"
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]+abs(c)*(phiold[(j+1)%nx]-phiold[j])

        phiold = phi.copy()
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)
    return phiold , M , V , TV

def BTCS( x , t , phiold , c ):
    "BTCS scheme. Input and output described at the beginning of the module"
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    # We are going to solve a linear system of equations, where the matrix of
    # coefficients is a circulant matrix, which is only defined by its first
    # column that we will represent as a vector named 'q' "
    q = np.zeros(nx)
    q[0] = 1
    q[1] = -c/2
    q[-1] = c/2
    for it in range(nt):
        phiold = linalg.solve_circulant(q,phiold)
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)
    return phiold , M , V , TV

def Lax_Wendroff( x , t , phiold , c ):
    "Lax Wendroff scheme. Input and output described at the beginning of the module"
    nx = len(x)
    nt = len(t)
    phi = np.zeros_like(phiold)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j]=phiold[j]-c/2*( (1-c)*phiold[(j+1)%nx] + 2*c*phiold[j] - (1+c)*phiold[(j-1)%nx] )
        phiold=phi.copy()
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)
    return phiold , M , V , TV

def Warming_Beam( x , t , phiold , c ):
    "Warming and Beam schem. Input and output described at the beginning of the module"
    nx = len(x)
    nt = len(t)
    phi = np.zeros_like(phiold)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)

    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]-c/2*( (3-c)*phiold[j] - 2*(2-c)*phiold[(j-1)%nx] + (1-c)*phiold[(j-2)%nx] )

        phiold = phi.copy()
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)
    return phiold , M , V , TV


def TVD( x , t , phiold , c ):
    "TVD scheme. Input and output described at the beginning of the module. "
    "The following code is valid only in the case c>0."
    "First we are going to evaluate the fluxes between cells, then we are going to"
    "evaluate the value of the solution in each cell as the difference between"
    "two consecutive fluxes."

    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    eps = 1e-15 # setting a tollerance to check if things are zero
    for it in range(nt):
        flux = np.zeros_like(phiold) # array containing the fluxes between cells
        for j in range(nx):
            if( np.abs( phiold[(j+1)%nx]-phiold[j] ) > eps) : #check if the gradient is not null
                # define the local gradient r
                r = ( phiold[j]-phiold[(j-1)%nx] )/( phiold[(j+1)%nx]-phiold[j] )
                # define the Superbee flux limiter
                psi = max(0,min(2*r,1),min(r,2))
                # defining the two fluxes, Lax Wendroff and upwind
                philax = (1+c)/2*phiold[j] + (1-c)/2*phiold[(j+1)%nx]
                phiup = phiold[j]
                # defining the total flux
                flux[j] = psi*philax + (1-psi)*phiup
            else:   #if gradient is zero, define flux as just upwind flux
                flux[j] = phiold[j]

        # Since pbc, in order to use np.diff, we define a new flux array which is
        # equal to the original one with the insertion of new element at the beginning,
        # equal to the last flux. With np.diff we can now evaluate how the value of the solution
        # in the cells have changed.
        flux = np.insert(flux,0,flux[-1])
        phiold = phiold -c*np.diff(flux) # evaluating new values of the solution

        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)

    return phiold , M , V , TV
