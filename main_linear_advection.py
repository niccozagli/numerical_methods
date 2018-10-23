import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 1, 'dx' : 4*10**-4,
                'tmin': 0 , 'tmax' : 1, 'dt' : 2*10**-4,
                'fluid_velocity' : 0.2}
    linearAdvect(parameters)

def linearAdvect(parameters):
    "Input parameters"
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    dx = parameters['dx']
    dt = parameters['dt']
    u = parameters['fluid_velocity']

    "Derived parameters"
    c = u/(dx/dt)
    print("The value of the Courant number is c={}.".format(c))
    x = np.arange(xmin,xmax,dx)
    t = np.arange(tmin,tmax,dt)

    "Initial condition"
    alpha = 0
    beta = 0.4
    phi0 = cosBell( x , alpha , beta )

    "Analytical solution at the last time tmax"
    phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta  )

    "Integration of the PDE using one of the advection schemes"
    "The BTCS scheme is unconditionally stable and conservative (if pbc)"
    phiBTCS , M , V = BTCS( x , t , phi0.copy() , c )

    "Plotting the results"
    plt.figure(1)
    plt.ion()
    plt.plot(x,phi0,'b',label='Initial Condition')
    plt.plot(x,phiAnalytic,'k',linestyle='--')
    plt.show()




#Calling the main function
main()
