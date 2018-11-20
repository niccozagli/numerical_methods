import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *

def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 1, 'x_sample_points' : 6e+2,
                'tmin': 0 , 'tmax' : 1, 't_sample_points' : 3e+2,
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

    "Initial condition"
    alpha = x[int(0.2*len(x))]
    beta = x[int(0.4*len(x))]
    phi0 = cosBell( x , alpha , beta )
    phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta )

    phi1 , M , V , TV = BTCS( x , t , phi0.copy() , c )
    phi2 , M , V , TV = Lax_Wendroff( x , t , phi0.copy() , c )
    phi3 , M , V , TV = Warming_Beam( x , t , phi0.copy() , c )
    phi4 , M , V , TV = TVD( x , t , phi0.copy() , c )

    plt.figure(0)
    plt.plot(x,phi0,'b--')
    plt.plot(x,phiAnalytic,'b')
    plt.plot(x,phi1,'kd',markersize=1)
    plt.plot(x,phi2,'ro',markersize=1)
    plt.plot(x,phi3,'g+',markersize=1)
    plt.plot(x,phi4,'yx',markersize=1)


    phi0 = squareWave( x , alpha , beta )
    phiAnalytic = squareWave( (x-u*tmax)%(xmax-xmin) , alpha , beta )

    phi1 , M , V , TV = BTCS( x , t , phi0.copy() , c )
    phi2 , M , V , TV = Lax_Wendroff( x , t , phi0.copy() , c )
    phi3 , M , V , TV = Warming_Beam( x , t , phi0.copy() , c )
    phi4 , M , V , TV = TVD( x , t , phi0.copy() , c )

    plt.figure(1)
    plt.plot(x,phi0,'b--')
    plt.plot(x,phiAnalytic,'b')
    plt.plot(x,phi1,'k')
    plt.plot(x,phi2,'r')
    plt.plot(x,phi3,'g')
    plt.plot(x,phi4,'yo',markersize=2)

    "Change nt to get a very high Courant number"
    tmax = 0.75
    nt = int(3e1)
    t = np.linspace(tmin,tmax,nt)
    dt = t[1]-t[0]
    c = u*dt/dx

    phiAnalytic = squareWave( (x-u*tmax)%(xmax-xmin) , alpha , beta )
    phi1 , M , V , TV = BTCS( x , t , phi0.copy() , c )
    phi2 , M , V , TV = Lax_Wendroff( x , t , phi0.copy() , c )
    phi3 , M , V , TV = Warming_Beam( x , t , phi0.copy() , c )
    phi4 , M , V , TV = TVD( x , t , phi0.copy() , c )

    plt.figure(2)
    plt.plot(x,phi0,'b--')
    plt.plot(x,phiAnalytic,'b')

    plt.plot(x,phi2,'r')
    plt.plot(x,phi3,'g')
    plt.plot(x,phi4,'yo',markersize=2)

    plt.figure(3)
    plt.plot(x,phi0,'b--')
    plt.plot(x,phiAnalytic,'b')
    plt.plot(x,phi1,'k')

    plt.show()


main()
