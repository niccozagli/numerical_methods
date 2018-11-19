import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import *
from AdvectionSchemes import *
from diagnostic import *

"Input parameters"
xmin = 0
xmax = 1.5
tmin = 0
tmax = 1.5
u = 0.2

nx_list = [5*i*1e1 for i in range(4,15)]
nt_list = [5*i*2e1 for i in range(4,15)]
error = np.zeros((len(nx_list),5))
dx_vec = np.zeros(len(nx_list))
alpha = 0.2
beta = 0.3



for i in range(len(nx_list)):
    "Derived parameters"
    x = np.linspace(xmin,xmax,int(nx_list[i]))
    phi0 = gaussian(x,mean=0.4,std=0.08)
    dx = x[1]-x[0]
    dx_vec[i] = dx
    t = np.linspace(tmin,tmax,int(nt_list[i]))
    dt = t[1]-t[0]
    c = u*dt/dx
    print(c)

    phiAnalytic = gaussian( (x-u*nt_list[i]*dt)%(xmax-xmin) , mean=0.4 , std=0.08 )
    phi , _ , _ , _ = upwind( x , t , phi0.copy() , c )
    error[i][0] = lpErrorNorm( phi , phiAnalytic , p=2 )
    #phi , _ , _ , _ = BTCS( x , t , phi0.copy() , c )
    #error[i][1] = lpErrorNorm( phi , phiAnalytic , p=2 )
    phi , _ , _ , _ = Lax_Wendroff( x , t , phi0.copy() , c )
    error[i][2] = lpErrorNorm( phi , phiAnalytic , p=2 )
    phi , _ , _ , _ = Warming_Beam( x , t , phi0.copy() , c )
    error[i][3] = lpErrorNorm( phi , phiAnalytic , p=2 )
    phi , _ , _ , _ = TVD( x , t , phi0.copy() , c )
    error[i][4] = lpErrorNorm( phi , phiAnalytic , p=2 )
    #plt.plot(x,phi,'k')
    #plt.show()

#plt.plot(dx_vec,f(dx_vec))
plt.plot(dx_vec,error[:,0],'rd')
plt.plot(dx_vec,error[:,2],'bx')
plt.plot(dx_vec,error[:,3],'ko')
plt.yscale("log")
plt.xscale("log")
plt.plot(dx_vec,np.exp(5.15)*dx_vec**2,'b--',linewidth=0.8)
plt.plot(dx_vec,np.exp(2.8)*dx_vec,'r--',linewidth=0.8)
plt.plot(dx_vec,np.exp(0.1)*dx_vec,'r--',linewidth=0.8)


plt.plot(dx_vec,error[:,1],'bo')
#plt.plot(dx_vec,error[:,2],'go')

plt.plot(dx_vec,error[:,4],'yo')
#plt.plot(dx_vec,dx_vec)
#plt.plot(dx_vec,dx_vec**2)
#plt.yscale("log")
#plt.xscale("log")
plt.show()
