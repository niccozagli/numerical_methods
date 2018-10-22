import numpy as np
#Starting to write code in python
def main():
    "Setting up the parameters of the integration and the fluid"
    "Integrating the linear advection equation, starting from a given"
    "initial condition"
    parameters={'xmin': 0 , 'xmax' : 2, 'dx' : 4*10**-4,
                'tmin': 0 , 'tmax' : 3, 'dt' : 2*10**-4,
                'fluid_velocity' : 0.9}
    linearAdvect(parameters)

def linearAdvect(parameters):
    xmin = parameters['xmin']
    xmax = parameters['xmax']
    tmin = parameters['tmin']
    tmax = parameters['tmax']
    dx = parameters['dx']
    dt = parameters['dt']
    u = parameters['fluid_velocity']
    #Derived parameters
    c = u/(dx/dt)
    print("The value of the Courant number is c={}.".format(c))
    x = np.arange(xmin,xmax,dx)
    t = np.arange(tmin,tmax,dt)


#Calling the main function
main()
