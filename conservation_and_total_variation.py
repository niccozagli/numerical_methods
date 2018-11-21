# This module contains functions that evaluate the moments and the total variation
# of numerical solution of the PDE
import numpy as np

def first_mom( phi ):
    "This function returns the first moment of phi"
    return sum(phi)

def second_mom( phi ):
    "This function returns the second moment of phi"
    return sum(phi**2)

def tot_var( phi ):
    "This function returns the total variation of phi"
    return sum( np.abs( np.diff(phi) ) )
