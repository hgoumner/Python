# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:44:00 2018

This file generates a regular, rectangular circle arrangement in a box.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import matplotlib for plotting
import matplotlib.pyplot as plt

#%% Generate circles in square

# circle radius
r = 6

# sqrt(number of circles)
n = 3

# total circle area
ac = n*n*np.pi*r**2

# square edge length
el = 64

# square area
asq = el**2

# area/volume percent
aper = 100*ac/asq

# distance between each circle for rectangular arrangement of circles
d = (el-n*2*r)/(n+1)

# coordinates of circle centers
x = np.arange(d+r,n*(2*r+d),2*r+d)
y = np.arange(d+r,n*(2*r+d),2*r+d)

# assign 2D array for circle centers
X,Y = np.meshgrid(x,y)
xf = X.flatten()
yf = Y.flatten()

#%% Plot circles

fig, ax = plt.subplots()

plt.scatter(xf,yf,color='b')
for i in range(len(xf)):
    circle = plt.Circle((xf[i], yf[i]), r, edgecolor='b', fill=False)
    ax.add_artist(circle)
plt.xlim([0,el])
plt.ylim([0,el])

#%% Export data

#scale = 10**(-6)
#out = np.c_[xf,yf,np.tile(r,(n*n,1))]*scale
#np.savetxt('xyr.csv', out, fmt='%10.10f', delimiter=',')
