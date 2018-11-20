from initial_conditions import *
from AdvectionSchemes import *
from conservation_and_total_variation import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def get_moments_figure():
    parameters={'xmin': 0 , 'xmax' : 1, 'x_sample_points' : 3e+2,
                'tmin': 0 , 'tmax' : 1, 't_sample_points' : 6e+2,
                'fluid_velocity' : 0.4}
    mom(parameters)



def mom(parameters):
    "Input parameters"
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    nx = int(parameters['x_sample_points'])
    nt = int(parameters['t_sample_points'])
    u = parameters['fluid_velocity']

    "Derived parameters"
    x = np.linspace(xmin,xmax,nx,endpoint=False)
    t = np.linspace(tmin,tmax,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    c = u*dt/dx
    print(c)

    "Square Wave as initial condition"

    alpha = 0.05
    beta = 0.35
    phi0 = squareWave( x , alpha , beta )
    phiAnalytic = squareWave( (x-u*tmax)%(xmax-xmin) , alpha , beta )

    "Integration of the PDE"
    phiup , Mup , Vup , TVup = upwind( x , t , phi0.copy() , c )
    phiBTCS , MBTCS , VBTCS , TVBTCS = BTCS( x , t , phi0.copy() , c )
    phiLax , MLax , VLax , TVLax = Lax_Wendroff( x , t , phi0.copy() , c )
    phiWar , MWar , VWar , TVWar = Warming_Beam( x , t , phi0.copy() , c )
    phiTVD , MTVD , VTVD , TVTVD = TVD( x , t , phi0.copy() , c )

    "Plots"

    plt.figure(0)
    tau = 1
    ms = 3.5
    lw = 1.5
    plt.figure(0)
    plt.plot(x,phi0,'k--')
    plt.plot(x,phiAnalytic,'k',linewidth=2)
    plt.plot(x[::tau],phiup[::tau],'bd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiBTCS[::tau],'cd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiLax[::tau],'ro-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiWar[::tau],'g+-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiTVD[::tau],'mx-',markersize=ms,linewidth=lw)
    plt.show()

    plt.figure(1)
    tau = 18
    plt.plot(t[::tau],Vup[::tau],'d')
    plt.plot(t[::tau],VBTCS[::tau],'rd')
    plt.plot(t[::tau],VLax[::tau],'go')
    plt.plot(t[::tau],VWar[::tau],'c+')
    plt.plot(t[::tau],VTVD[::tau],'mx')
    plt.show()

    plt.figure(2)
    tau = 20
    ms = 4.5
    plt.plot(t[::tau],TVup[::tau],'d',markersize=ms)
    plt.plot(t,TVBTCS,'r')
    plt.plot(t,TVLax,'g')
    plt.plot(t,TVWar,'c')
    plt.plot(t,TVTVD,'m')
    plt.show()
