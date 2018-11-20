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

nx_list = [5*i*1e1 for i in range(4,10)]
nt_list = [5*i*2e3 for i in range(4,10)]
error = np.zeros(len(nx_list))
dx_vec = np.zeros(len(nx_list))

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
    phi , _ , _ , _ = BTCS( x , t , phi0.copy() , c )
    error[i] = lpErrorNorm( phi , phiAnalytic , p=2 )

plt.plot(dx_vec,error)
plt.plot(dx_vec,np.exp(5.15)*dx_vec**2,'b--',linewidth=0.8)
plt.yscale("log")
plt.xscale("log")
plt.show()

plt.figure(2)
plt.plot(x,phi,'bo')
plt.plot(x,phiAnalytic)
plt.show()
