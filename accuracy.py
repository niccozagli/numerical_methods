# This module contains the definition of functions that evaluate how the
# error between the numerical and exact solutions is related to the uniform spatial
# grid resolution.

# Importing useful modules
import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *
from diagnostic import *

# Definition of the function called in main_final.py
def get_accuracy_figure():
    "This function does not have input. It simply defines the parameters of the"
    "integration corresponding to multiple sets of resolution in the spatial"
    "and temporal grid"

    # Definition of the parameters of the integration. The number of sample points
    # is not specified in parameters.
    parameters={'xmin': 0 , 'xmax' : 0.9, 'x_sample_points' : np.NaN ,
                'tmin': 0 , 'tmax' : 2, 't_sample_points' : np.NaN ,
                'fluid_velocity' : 0.1}

    # Definition of different sets of resolution. These have been chosen in order
    # to get a good sampling of small enough spatial resolutions. Observe that the
    # temporal resolution is much smaller than the spatial one.
    nx_list = [5*i*1e1 for i in range(4,9)]
    nt_list = [1*i*1e3 for i in range(4,9)]
    acc(parameters,nx_list,nt_list)


# Definitio of the second function
def acc(parameters,nx_list,nt_list):
    "This function integrates the PDE multiple times. Each time it evaluates the"
    "error between the numerical and exact solution. This function has 3 inputs:"
    "parameters set all the parameters for the integration but the spatial and"
    "temporal grid spacing. The other two inputs nx_list and nt_list give a set of "
    "spatial and temporal grid spacing to integrate the PDE with. The result of the"
    "function is plotting the results in the Figure directory."

    # Input parameters
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    u = parameters['fluid_velocity']

    # Defining a matrix error. The rows are corresponding to different settings of
    # the integration, the columns to the different schemes.
    # Also defining a vector dx_vec to store the value of the spatial resolutions
    error = np.zeros( (len(nx_list),5) )
    dx_vec = np.zeros(len(nx_list))

    print("There will be an iteration on a total number of {} different sets of resolution \n".format(len(nx_list)))

    for i in range(len(nx_list)): # loop on different resolutions
        print("Currently at iteration number {} \n".format(i+1))

        #Derived parameters
        x = np.linspace(xmin,xmax,int(nx_list[i]),endpoint=False)
        t = np.linspace(tmin,tmax,int(nt_list[i]))
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        dx_vec[i] = dx
        c = u*dt/dx

        # Using a smooth initial condition (Gaussian)
        mean = 0.4
        std = 0.1
        phi0 = gaussian(x,mean,std)
        phiAnalytic = gaussian( (x-u*nt_list[i]*dt)%(xmax-xmin) , mean , std )

        # Integrating the PDE with different schemes
        phiup , _ , _ , _ = upwind( x , t , phi0.copy() , c )
        error[i,0] = lpErrorNorm( phiup , phiAnalytic , p=2 )
        phiLax , _ , _ , _ = Lax_Wendroff( x , t , phi0.copy() , c )
        error[i,1] = lpErrorNorm( phiLax , phiAnalytic , p=2 )
        phiWar , _ , _ , _ = Warming_Beam( x , t , phi0.copy() , c )
        error[i,2] = lpErrorNorm( phiWar , phiAnalytic , p=2 )
        phiTVD , _ , _ , _ = TVD( x , t , phi0.copy() , c )
        error[i,3] = lpErrorNorm( phiTVD , phiAnalytic , p=2 )
        phibtcs , _ , _ , _ = BTCS( x , t , phi0.copy() , c )
        error[i,4] = lpErrorNorm( phibtcs , phiAnalytic , p=2 )

    # Plotting the results
    plt.figure(0)
    plt.clf()
    plt.plot(dx_vec,error[:,0],'bd')
    plt.plot(dx_vec,error[:,1],'ro')
    plt.plot(dx_vec,error[:,2],'g+')
    plt.plot(dx_vec,error[:,3],'mx')
    plt.plot(dx_vec,error[:,4],'cd')

    # Plotting two lines with known slopes and suitable offset and using log scale
    plt.plot(dx_vec,np.exp(4.1)*dx_vec**2,'--')
    plt.plot(dx_vec,np.exp(2.1)*dx_vec,'r--')
    plt.yscale("log")
    plt.xscale("log")
    ax = plt.gca()
    ax.set_xlabel( r"$\Delta x$" , labelpad = 0.1 )
    ax.set_ylabel( r"$l_2(t_{max})$" , labelpad = 0.1 )
    plt.savefig('Figures/Accuracy.eps', dpi = 1000 )
