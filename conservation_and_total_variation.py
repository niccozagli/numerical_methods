import numpy as np
def first_mom( phi ):
    return sum(phi)

def second_mom( phi ):
    return sum(phi**2)

def tot_var( phi ):
    return sum(np.abs(np.diff(phi)))
