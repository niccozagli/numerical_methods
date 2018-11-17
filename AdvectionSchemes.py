"In this module we will define some advection schemes to integrate the linear"
"advection equation. In particular, we have defined two integration schemes: "
" - the upwind scheme (which is FTBS if the velocity 'u' of the fluid is u>=0"
" and FTFS if u<0"
" - the BTCS scheme."

"Each scheme returns the evoluted variable at the last time tmax and the whole"
"temporal evolution of the first 'M' and second 'V' moment of the variable "

"Recall : the schemes are valid for a system with periodic boundary conditions"

import numpy as np
import scipy.linalg as linalg
from conservation_and_total_variation import *

def upwind( x , t , phiold , c ):
    if(c>=0):
        phi_upwind , M , V , TV = FTBS( x , t , phiold , c )
    else:
        phi_upwind , M , V , TV = FTFS(x , t , phiold , c )
    return phi_upwind , M , V

def FTBS( x , t , phiold , c ):
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
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    "We are going to solve a linear system of equations, where the matrix of "
    "coefficients is a circulant matrix, which is only defined by its first"
    "column that we will represent as a vector named 'q' "
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
    # This works if c>0
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    TV = np.zeros(nt)
    eps = 1e-18
    for it in range(nt):
        flux = np.zeros_like(phiold)
        for j in range(nx):
            if( np.abs( phiold[(j+1)%nx]-phiold[j] ) > eps) :
                r = ( phiold[j]-phiold[(j-1)%nx] )/( phiold[(j+1)%nx]-phiold[j] )
                psi = max(0,min(2*r,1),min(r,2))#(r**2+r)/(r**2+1)#( r+np.abs(r) )/( 1+np.abs(r) )
                philax = (1+c)/2*phiold[j] + (1-c)/2*phiold[(j+1)%nx]
                phiup = phiold[j]
                flux[j] = psi*philax + (1-psi)*phiup
            else:
                flux[j] = phiold[j]

        flux = np.insert(flux,0,flux[-1])
        phiold = phiold -c*np.diff(flux)

        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
        TV[it] = tot_var(phiold)

    return phiold , M , V , TV
