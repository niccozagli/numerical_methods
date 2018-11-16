import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from initial_conditions import *
from AdvectionSchemes import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 1, 'dx' : 0.5e-3,
                'tmin': 0 , 'tmax' : 1, 'dt' : 0.2e-3,
                'fluid_velocity' : 0.3}
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
    c = u*dt/dx
    print("The value of the Courant number is c={}.".format(c))
    x = np.arange(xmin,xmax,dx)
    t = np.arange(tmin,tmax,dt)

    "Initial condition"
    alpha = 0.1
    beta = 0.3
    #phi0 = cosBell( x , alpha , beta )
    phi0 = squareWave(x,alpha,beta)
    "Analytical solution at the last time tmax"
    #phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta )
    phiAnalytic = squareWave( (x-u*tmax)%(xmax-xmin) , alpha , beta )
    "Integration of the PDE using one of the advection schemes"
    "The BTCS scheme is unconditionally stable and conservative (if pbc)"
    #phi , M , V , TV = BTCS( x , t , phi0.copy() , c )
    #phi , M , V , TV = Lax_Wendroff( x , t , phi0.copy() , c )
    #phi , M , V , TV = Warming_Beam( x , t , phi0.copy() , c )
    phi , M , V , TV = TVD( x , t , phi0.copy() , c )
    "Plotting the results"
    plt.figure(0)
    plt.clf()
    plt.plot(x,phi0,'b',label='Initial Condition')
    plt.plot(x,phiAnalytic,'k',linestyle='--',label='Analytical solution')
    plt.plot(x,phi,'r',label='Numerical solution',linewidth=0.4)
    #plt.legend()
    plt.xlabel('position x')
    plt.ylabel('Independent variable')
    plt.title('Initial, analytical and numerical solution')
    plt.savefig('./comparison.pdf')
    plt.show()

    plt.figure(1)
    plt.clf()
    plt.plot(t[1:],np.diff(M))
    plt.xlabel('time t')
    plt.title('M[t+1]-M[t] vs time')
    plt.savefig('./First_moment.pdf')
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.plot(t[1:],np.diff(V))
    plt.title('V[t+1]-V[t] vs time')
    plt.savefig('./Second_moment.pdf')
    plt.show()

    plt.figure(3)
    plt.plot(t,TV)
    plt.show()


"Calling the main function of the program"
main()
