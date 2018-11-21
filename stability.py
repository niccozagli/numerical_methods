# In this module two functions are defined in order to get the following
# results.
#1) A plot of the numerical integrated solution of the linear advection
# equation using a continuous initial condition.
# 2) Same plot when the numerical schemes are unstable.
# 3) Plot of the numerical solution using the BTCS scheme, which is always
# stable.
# 4) Plot of the first moment

# Importing modules
from initial_conditions import *
from AdvectionSchemes import *
from conservation_and_total_variation import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Defining the first function, called in the main script "main_final.py"
def get_stability_figure():
    "This function is called in the main file. It doesn't have any input and the"
    "only result is saving figures in the directory Figures"

    # Setting up all the parameters of the integration.
    parameters = {'xmin': 0 , 'xmax' : 1, 'x_sample_points' : 2e+2 ,
                'tmin': 0 , 'tmax' : 0.75, 't_sample_points' : 5e+2 ,
                'fluid_velocity' : 0.3}
    # Calling the function "integrate" with the flag "saveBTCSfig" set to zero.
    # The parameters have been chosen in order to have all the numerical
    # schemes stable
    integrate( parameters , saveBTCSfig=0 )

    # Using a different set of parameters to make the all the numerical schemes,
    # but the BTCS, unstable. Calling the function with the flag "saveBTCS" set
    # to 1.
    parameters['x_sample_points'] = 8e+2
    parameters['t_sample_points'] = 3e+1
    integrate( parameters , saveBTCSfig=1 )


# Defining the second function of the module
def integrate(parameters,saveBTCSfig):
    "This function has two inputs. parameters is a dictionary containing all the"
    "parameters for the integration. The second input is a flag: set to zero, the"
    "function saves a plot of the solution corresponding to the stable"
    "configuration and a plot of the corresponing first moments. Set to 1, the "
    "it saves a plot when the numerical schemes are unstable and also a separate"
    "plot for the unconditionally stable BTCS scheme."

    #Input parameters
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    nx = int(parameters['x_sample_points'])
    nt = int(parameters['t_sample_points'])
    u = parameters['fluid_velocity']

    #Derived parameters
    x = np.linspace(xmin,xmax,nx,endpoint=False) #set endpoint=False to implement periodic boundary conditions
    t = np.linspace(tmin,tmax,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    c = u*dt/dx


    #Using a continuous cosine Bell as initial condition
    alpha = 0.2
    beta = 0.5
    phi0 = cosBell( x , alpha , beta )
    phiAnalytic = cosBell( (x-u*tmax)%(xmax-xmin) , alpha , beta )

    #Integration the linear advection equation with different schemes
    phiup , Mup , _ , _ = upwind( x , t , phi0.copy() , c )
    phiBTCS , MBTCS , _ , _ = BTCS( x , t , phi0.copy() , c )
    phiLax , MLax , _ , _ = Lax_Wendroff( x , t , phi0.copy() , c )
    phiWar , MWar , _ , _ = Warming_Beam( x , t , phi0.copy() , c )
    phiTVD , MTVD , _ , _ = TVD( x , t , phi0.copy() , c )

    #Plotting the graph of the analytical and numerical solutions
    plt.rc('text', usetex=True)
    plt.rc('font',size=16)
    tau = 2
    ms = 3.5
    lw = 0.5
    plt.figure(0)
    plt.clf()
    plt.plot(x,phi0,'k--')
    plt.plot(x,phiAnalytic,'k',linewidth=2)
    plt.plot(x[::tau],phiup[::tau],'bd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiBTCS[::tau],'cd-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiLax[::tau],'ro-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiWar[::tau],'g+-',markersize=ms,linewidth=lw)
    plt.plot(x[::tau],phiTVD[::tau],'mx-',markersize=ms,linewidth=lw)
    plt.xlim((0.1,0.9))
    ax = plt.gca()
    ax.set_xlabel( r"$x$" , labelpad=0.1 )
    ax.set_ylabel(r"$\varphi$")


    if( saveBTCSfig == 0 ): #Inside this block, the configuration is stable
        # Saving figure 0 in the directory Figures
        plt.savefig('Figures/Stable_configuration.eps', dpi = 1000 )
        # Plotting the first moment
        plt.figure(1)
        plt.clf()
        tau = 15
        plt.plot(t[::tau],Mup[::tau],'b')
        plt.plot(t[::tau],MBTCS[::tau],'rd')
        plt.plot(t[::tau],MLax[::tau],'go')
        plt.plot(t[::tau],MWar[::tau],'c+')
        plt.plot(t[::tau],MTVD[::tau],'mx')
        plt.ylim((29.9,30.1))
        ax1 = plt.gca()
        ax1.set_xlabel(r"$t$", labelpad=0.1 )
        ax1.set_ylabel(r"$M(t)$", labelpad=0.1 )
        # Saving figure 1 in the directory Figures
        plt.savefig('Figures/First_moments.eps', dpi = 1000 )

    elif( saveBTCSfig==1 ): #In this block the configuration is unstable
        # Saving figure 0 in the directory Figures
        plt.savefig('Figures/Unstable_configuration.eps', dpi = 1000 )
        # Plotting the figure for the BTCS
        plt.figure(1)
        plt.clf()
        tau = 2
        plt.plot(x,phi0,'k--')
        plt.plot(x,phiAnalytic,'k',linewidth=2)
        plt.plot(x[::tau],phiBTCS[::tau],'cd-',markersize=ms,linewidth=lw)
        plt.xlim((0.1,0.9))
        ax1 = plt.gca()
        ax1.set_xlabel( r"$x$" , labelpad=0.1 )
        ax1.set_ylabel(r"$\varphi$")
        plt.savefig('Figures/BTCS_scheme.eps', dpi = 1000 )
