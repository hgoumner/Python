# -*- coding: utf-8 -*-
"""

This file computes the average value of a given parameter for a circular area.

@author: Hristo Goumnerov
"""
#%% Import Modules

# import numpy for advanced mathematical operations
import numpy as np

# import matplotlib for plotting
import matplotlib.pyplot as plt

#%% Code

# Data inputs
fn = "R:/166/TRANSFER/Goumnerov/TOM_wave/Simulation/_Current/Temperature Intensity 1.csv"

# Load Data
data = np.loadtxt(fn)

# Assign data to variables
x = data[:,0]
y = data[:,1]

# Determine integration limits
xmin = min(x)
xmax = max(x)

# Integration area
A_tot = np.pi*xmax**2 #4*trapz(sqrt(xmax**2-x**2),x)

# Compute trapezoidal integration integral of 2*pi*T*r*dr within given limits 
T_int = np.trapz(2*np.pi*y*x,x)

# Compute average value of variable
T_bar = T_int/A_tot

#%% Plot

# Replicate variable for plotting
T_bar2 = T_bar*np.ones([len(x),1]) 

# Create plot of input and output
plt.plot(x,y,'r',x,T_bar2,'b')
plt.legend(["Input Data","Average Value"])
plt.ylim([T_bar-(max(y)-T_bar)-50,max(y)+50])
plt.xlabel('r')
plt.ylabel('T')