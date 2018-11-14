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
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
    return phiold , M , V

def FTFS( x , t , phiold , c ):
    phi = np.zeros_like(phiold)
    nx = len(x)
    nt = len(t)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j] = phiold[j]+abs(c)*(phiold[(j+1)%nx]-phiold[j])

        phiold = phi.copy()
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
    return phiold , M , V

def BTCS( x , t , phiold , c ):
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
        phiold = linalg.solve_circulant(q,phiold)
        M[it] = first_mom(phiold)
        V[it] = second_mom(phiold)
    return phiold , M , V

def Lax_Wendroff( x , t , phiold , c ):
    nx = len(x)
    nt = len(t)
    phi = np.zeros_like(phiold)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j]=phiold[j]-c/2*( (1-c)*phiold[(j+1)%nx] + 2*c*phiold[j] - (1+c)*phiold[(j-1)%nx] )
        phiold=phi.copy()
        M[it]=first_mom(phiold)
        V[it]=second_mom(phiold)
    return phiold , M , V

def Warming_Beam( x , t , phiold , c ):
    nx = len(x)
    nt = len(t)
    phi = np.zeros_like(phiold)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            phi[j]=phiold[j]-c/2*( (3-c)*phiold[j] - 2*(2-c)*phiold[(j-1)%nx] + (1-c)*phiold[(j-2)%nx] )
        phiold=phi.copy()
        M[it]=first_mom(phiold)
        V[it]=second_mom(phiold)
    return phiold , M , V


def TVD( x , t , phiold , c ):
    if(c>=0):
        phi , M , V = TVD_positive_speed( x , t , phiold , c )
    else:
        phi , M , V = TVD_negative_speed( x , t , phiold, c)
    return phi , M , V


def TVD_positive_speed( x , t , phiold , c):
    nx = len(x)
    nt = len(t)
    phi = np.zeros_like(phiold)
    M = np.zeros(nt)
    V = np.zeros(nt)
    for it in range(nt):
        for j in range(nx):
            A = phiold[j] - phiold[(j-1)%nx]
            B = phiold[(j+1)%nx] - phiold[j]
            C = phiold[(j-1)%nx] - phiold[(j-2)%nx]
            eps = 10**-13
            if( np.abs(A)>eps and np.abs(B)>eps):
                r_right = A/B
                r_left = C/A
                psi_right = (r_right+np.abs(r_right))/(1+np.abs(r_right))
                psi_left = (r_left+np.abs(r_left))/(1+np.abs(r_left))
            else:
                if(np.abs(B)<eps and np.abs(A)>eps):
                    psi_right = 1+np.sign(A)
                    r_left = C/A
                    psi_left = (r_left+np.abs(r_left))/(1+np.abs(r_left))
                elif(np.abs(B)>eps and np.abs(A)<eps):
                    r_right = 0
                    psi_right = 0
                    psi_left = 1+np.sign(C)
                elif(np.abs(B)<eps and np.abs(A)<eps):
                    # r_right = 1
                    psi_right = 1
                    psi_left =  1+np.sign(C)

            phi_Hr = (1+c)/2*phiold[j]+(1-c)/2*phiold[(j+1)%nx]
            phi_Lr = phiold[j]
            phi_right = psi_right*phi_Hr + (1-psi_right)*phi_Lr

            phi_Hl = (1+c)/2*phiold[(j-1)%nx]+(1-c)/2*phiold[j]
            phi_Ll = phiold[(j-1)%nx]
            phi_left = psi_left*phi_Hl + (1-psi_left)*phi_Ll

            phi[j] = phiold[j] - c*(phi_right-phi_left)
        phiold=phi.copy()
        M[it]=first_mom(phiold)
        V[it]=second_mom(phiold)
    return phiold , M , V
