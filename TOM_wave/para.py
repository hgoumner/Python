# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 08:58:41 2018

This file reads in data and sorts it by X and then by Y so that it can be 
plotted in ParaView.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

#%% Sort and export data

# assuming data.csv is a CSV file with the 1st row being the names names for
# the columns

# filename
fn = "R:/166_320/Transfer/Goumnerov/zWiMi/GAUSS.CSV"

# load file
data = np.loadtxt(fn, skiprows=1, delimiter=';')

# sort data
inds = np.lexsort((data[:,1],data[:,2]))
out = data[inds]

# export data
np.savetxt(fn[:-4]+'.csv',out)