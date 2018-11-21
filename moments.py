# This module contains function that integrate the linear advection equation starting with a
# discontinuous initial condition. As a result, it prints the following plot
# 1) Numerical solution of the linear adv equation given by different schemes
# 2) Temporal evolution of the second moment of the numerical solution
# 3) Temporal evolution of the total variation of the numerical solution

# Two functions are defined in this module. get_moments_figure is called in the
# main script main_final.py .

# Importing useful modules
from initial_conditions import *
from AdvectionSchemes import *
from conservation_and_total_variation import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Definition of the first function
def get_moments_figure():
    "This function is called in main_final.py . It doesn't have inputs and the "
    "only result is saving some figures in the directory Figures"

    # Setting the parameters of the integration. The parameters have been chosen
    # in order to get a good visualisation of how different numerical schemes
    # advect discontinuities.
    parameters = {'xmin': 0 , 'xmax' : 1, 'x_sample_points' : 3e+2,
                'tmin': 0 , 'tmax' : 1, 't_sample_points' : 6e+2,
                'fluid_velocity' : 0.4}
    # Calling the second function
    mom(parameters)

# Defining the second function
def mom(parameters):
    "This function has the dictionary parameters as an input, setting the parameters"
    "of the integration. The function then integrates the PDE using different schemes"

    # Input parameters
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    nx = int(parameters['x_sample_points'])
    nt = int(parameters['t_sample_points'])
    u = parameters['fluid_velocity']

    # Derived parameters
    x = np.linspace(xmin,xmax,nx,endpoint=False) # endpoint=False to get periodic boundary conditions
    t = np.linspace(tmin,tmax,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    c = u*dt/dx

    # Using a discontinuous square Wave as initial condition"
    alpha = 0.05
    beta = 0.35
    phi0 = squareWave( x , alpha , beta )
    phiAnalytic = squareWave( (x-u*tmax)%(xmax-xmin) , alpha , beta )

    # Integration of the PDE using different schemes
    phiup , _ , Vup , TVup = upwind( x , t , phi0.copy() , c )
    phiBTCS , _ , VBTCS , TVBTCS = BTCS( x , t , phi0.copy() , c )
    phiLax , _ , VLax , TVLax = Lax_Wendroff( x , t , phi0.copy() , c )
    phiWar , _ , VWar , TVWar = Warming_Beam( x , t , phi0.copy() , c )
    phiTVD , _ , VTVD , TVTVD = TVD( x , t , phi0.copy() , c )

    # Plotting and saving the results
    plt.rc('text', usetex=True)
    plt.rc('font',size=16)

    plt.figure(0)
    plt.clf()
    tau = 1
    ms = 3.5
    lw = 1.5
    plt.plot(x,phi0,'k--')
    plt.plot(x,phiAnalytic,'k',linewidth=2)
    plt.plot(x[::tau],phiup[::tau],'bd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiBTCS[::tau],'cd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiLax[::tau],'ro-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiWar[::tau],'g+-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiTVD[::tau],'mx-',markersize=ms,linewidth=lw)
    ax = plt.gca()
    ax.set_xlabel( r"$x$" , labelpad=0.1 )
    ax.set_ylabel( r"$\varphi$" , labelpad=0.1 )
    plt.savefig('Figures/Discontinuos_advection.eps', dpi = 1000 )

    plt.figure(1)
    plt.clf()
    tau = 18
    plt.plot(t[::tau],Vup[::tau],'d')
    plt.plot(t[::tau],VBTCS[::tau],'rd')
    plt.plot(t[::tau],VLax[::tau],'go')
    plt.plot(t[::tau],VWar[::tau],'c+')
    plt.plot(t[::tau],VTVD[::tau],'mx')
    ax1 = plt.gca()
    ax1.set_xlabel( r"$t$" , labelpad=0.1 )
    ax1.set_ylabel( r"$V(t)$", labelpad=0.1 )
    plt.savefig('Figures/Variance.eps', dpi = 1000 )

    plt.figure(2)
    plt.clf()
    tau = 20
    ms = 4.5
    plt.plot(t[::tau],TVup[::tau],'d',markersize=ms)
    plt.plot(t,TVBTCS,'r')
    plt.plot(t,TVLax,'g')
    plt.plot(t,TVWar,'c')
    plt.plot(t,TVTVD,'m')
    ax2 = plt.gca()
    ax2.set_xlabel( r"$t$" , labelpad=0.1 )
    ax2.set_ylabel( r"$TV(t)$", labelpad=0.1 )
    plt.savefig('Figures/Total_variation.eps', dpi = 1000 )
