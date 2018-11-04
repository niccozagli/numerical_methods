import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from initial_conditions import *
from AdvectionSchemes import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 2.5, 'dx' : 1*10**-4,
                'tmin': 0 , 'tmax' : 1.5, 'dt' : 4*10**-4,
                'fluid_velocity' : 0.8}
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
    plt.ion()
    plt.figure(0)
    plt.clf()
    plt.plot(x,phi0,'b',label='Initial Condition')
    plt.plot(x,phiAnalytic,'k',linestyle='--',label='Analytical solution')
    plt.plot(x,phiBTCS,'r',label='Numerical solution',linewidth=0.4)
    plt.legend()
    plt.xlabel('position x')
    plt.ylabel('Independent variable')
    plt.title('Initial, analytical and numerical solution')
    plt.savefig('./comparison.pdf')


    plt.figure(1)
    plt.clf()
    plt.plot(t[1:],np.diff(M))
    plt.xlabel('time t')
    plt.title('M[t+1]-M[t] vs time')
    plt.savefig('./First_moment.pdf')

    plt.figure(2)
    plt.clf()
    plt.plot(t[1:],np.diff(V))
    plt.title('V[t+1]-V[t] vs time')
    plt.savefig('./Second_moment.pdf')

    plt.show()

"Calling the main function of the program"
main()
