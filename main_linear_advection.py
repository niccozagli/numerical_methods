import numpy as np
from initial_conditions import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 2, 'dx' : 4*10**-4,
                'tmin': 0 , 'tmax' : 3, 'dt' : 2*10**-4,
                'fluid_velocity' : 0.9}
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
    beta = 0.6
    phi0 = cosBell( x , alpha , beta )

    "Analytical solution at the last time tmax"

    phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta  )





#Calling the main function
main()
