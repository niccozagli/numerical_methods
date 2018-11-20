import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *
from diagnostic import *

def get_accuracy_figure():

    parameters={'xmin': 0 , 'xmax' : 0.9, 'x_sample_points' : np.NaN ,
                'tmin': 0 , 'tmax' : 2, 't_sample_points' : np.NaN ,
                'fluid_velocity' : 0.1}

    nx_list = [5*i*1e1 for i in range(5,15,2)]
    nt_list = [5*i*2e1 for i in range(5,15,2)]
    #acc(parameters,nx_list,nt_list,btcsflag=0)

    nx_list = [5*i*1e1 for i in range(4,10)]
    nt_list = [1*i*2e3 for i in range(4,10)]
    acc(parameters,nx_list,nt_list,btcsflag=1)



def acc(parameters,nx_list,nt_list,btcsflag):
    "Input parameters"
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    u = parameters['fluid_velocity']

    error = np.zeros( (len(nx_list),4) )
    errorbtcs = np.zeros(len(nx_list) )
    dx_vec = np.zeros(len(nx_list))
    print("There will be an iteration on a total number of {} different sets of resolution \n".format(len(nx_list)))
    for i in range(len(nx_list)):
        print("Currently at iteration number {} \n".format(i+1))
        "Derived parameters"
        x = np.linspace(xmin,xmax,int(nx_list[i]),endpoint=False)
        t = np.linspace(tmin,tmax,int(nt_list[i]))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        dx_vec[i] = dx
        c = u*dt/dx
        print(c)

        "Initial condition"
        mean = 0.4
        std = 0.1
        phi0 = gaussian(x,mean,std)
        phiAnalytic = gaussian( (x-u*nt_list[i]*dt)%(xmax-xmin) , mean , std )

        if(btcsflag==0):
            phiup , _ , _ , _ = upwind( x , t , phi0.copy() , c )
            error[i][0] = lpErrorNorm( phiup , phiAnalytic , p=2 )
            phiLax , _ , _ , _ = Lax_Wendroff( x , t , phi0.copy() , c )
            error[i][1] = lpErrorNorm( phiLax , phiAnalytic , p=2 )
            phiWar , _ , _ , _ = Warming_Beam( x , t , phi0.copy() , c )
            error[i][2] = lpErrorNorm( phiWar , phiAnalytic , p=2 )
            phiTVD , _ , _ , _ = TVD( x , t , phi0.copy() , c )
            error[i][3] = lpErrorNorm( phiTVD , phiAnalytic , p=2 )
        elif(btcsflag==1):
            phibtcs , _ , _ , _ = BTCS( x , t , phi0.copy() , c )
            errorbtcs[i] = lpErrorNorm( phibtcs , phiAnalytic , p=2 )


    if(btcsflag==0):
        plt.plot(dx_vec,error[:,0],'bd')
        plt.plot(dx_vec,error[:,1],'ro')
        plt.plot(dx_vec,error[:,2],'g+')
        plt.plot(dx_vec,error[:,3],'mx')
        plt.plot(dx_vec,np.exp(4.5)*dx_vec**2,'b--',linewidth=0.8)
        plt.plot(dx_vec,np.exp(2.1)*dx_vec,'r--',linewidth=0.8)
    elif(btcsflag==1):
        plt.plot(dx_vec,errorbtcs,'cd')
        plt.plot(dx_vec,np.exp(3.9)*dx_vec**2,'--')


        plt.yscale("log")
        plt.xscale("log")
        plt.show()
