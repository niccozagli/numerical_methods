"In this module we will define some advection schemes to integrate the linear"
"advection equation. In particular, we have defined two integration schemes: "
" - the upwind scheme (which is FTBS if the velocity 'u' of the fluid is u>=0"
" and FTFS if u,0"
" - the BTCS scheme."

"Each scheme returns the evoluted variable at the last time tmax and the whole"
"temporal evolution of the first 'M' and second 'V' moment of the variable "

"Recall : the schemes are valid for a system with periodic boundary conditions"

import numpy as np
import scipy.linalg as linalg

def upwind( x , t , phiold , c ):
    if(c>=0):
        phi_upwind , M , V = FTBS( x , t , phiold , c )
    else:
        phi_upwind , M , V = FTFS(x , t , phiold , c )
    return phi_upwind , M , V

def FTBS( x , t , phiold , c ):
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]-c*(phiold[j%nx]-phiold[(j-1)%nx])

        phiold = phi.copy()
        M[it] = sum(phiold)
        V[it] = sum(phiold**2)
    return phiold , M , V

def FTFS( x , t , phiold , c ):
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]+abs(c)*(phiold[(j+1)%nx]-phiold[(j)%nx])

        phiold = phi.copy()
        M[it] = sum(phiold)
        V[it] = sum(phiold**2)
    return phiold , M , V

def BTCS( x , t , phiold , c ):
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    "We are going to solve a linear system of equations, where the matrix of "
    "coefficients is a circulant matrix, which is only defined by its first"
    "column that we will represent as a vector named 'q' "
    q = np.zeros(nx)
    q[0] = 1
    q[1] = -c/2
    q[-1] = c/2
    for it in range(nt):
        phi = linalg.solve_circulant(q,phiold)
        phiold = phi.copy()
        M[it] = sum(phiold)
        V[it] = sum(phiold**2)
    return phiold , M , V
