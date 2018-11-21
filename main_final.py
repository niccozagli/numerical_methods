# This is the main file. Compile this file to get all the results and figures
# needed for the report assignement.
# The code is organised in separated functions that return different figures
# This code has been developed to integrate the linear advection equation,
# with periodic boundary conditions, using different numerical schemes.
# Multiple features, such as stability, conservation of moments and total
# variation will be investigated.
# This script will create a directory named Figures in the current directory
# to save the figures in.

#Importing modules
from stability import *
from moments import *
from accuracy import *
import os

# Definition of the main function, that calls three functions contained in the
# imported modules.
def main():
    "This function creates a directory Figures and calls all the functions"
    "required to get the results for the report"
    dirName = "Figures"
    # Check if the directory already exists.
    try:
        os.mkdir(dirName)
        print("Directory", dirName, "created.\n")
    except FileExistsError:
        print("Directory", dirName, "already exists.\nStill going on with execution.\n")

    print("Printing figures concerning continuous initial condition. \n")
    get_stability_figure()

    print("Printing figures concerning discontinuous initial condition. \n")
    get_moments_figure()

    print("Moving on to accuracy tests. \n")
    get_accuracy_figure()


"Calling the main function"
main()
