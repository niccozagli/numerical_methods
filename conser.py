import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 1, 'x_sample_points' : 6e+2,
                'tmin': 0 , 'tmax' : 2, 't_sample_points' : 1e+3,
                'fluid_velocity' : 0.3}
    linearAdvect(parameters)

def linearAdvect(parameters):
    "Input parameters"
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    nx = int(parameters['x_sample_points'])
    nt = int(parameters['t_sample_points'])
    u = parameters['fluid_velocity']

    "Derived parameters"
    x = np.linspace(xmin,xmax,nx)
    t = np.linspace(tmin,tmax,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    c = u*dt/dx
    print("The value of the Courant number is c={}.".format(c))

    "Initial condition"
    alpha = x[int(0.2*len(x))]
    beta = x[int(0.4*len(x))]
    phi0 = cosBell( x , alpha , beta )
    #phi0 = squareWave(x,alpha,beta)
    "Analytical solution at the last time tmax"
    #phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta )
    phiAnalytic = squareWave( (x-u*nt*dt)%(xmax-xmin) , alpha , beta )
    "Integration of the PDE using one of the advection schemes"
    "The BTCS scheme is unconditionally stable and conservative (if pbc)"
    #phi , M , V , TV = upwind( x , t , phi0.copy() , c )
    #phi , M1 , V1 , TV1 = BTCS( x , t , phi0.copy() , c )
    #phi , M2 , V2 , TV2 = Lax_Wendroff( x , t , phi0.copy() , c )
    #phi , M3 , V3 , TV3 = Warming_Beam( x , t , phi0.copy() , c )
    phi , M4 , V4 , TV4 = TVD( x , t , phi0.copy() , c )
    "Plotting the results"
    plt.figure(0)
    plt.plot(t, TV4)
    plt.show()




"Calling the main function of the program"
main()
