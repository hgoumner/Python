# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:22:02 2017

This file loads a 3D array and interpolates a new 3D array using a spline fit.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy as sp
from scipy.ndimage import imread as imr

# import matplotlib for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import tkinter for file selection dialog
import tkinter
tkinter.Tk().withdraw()
from tkinter.filedialog import askopenfilename as ask

#%% Functions

# function to load image and convert it to a 3D integer array
def load_image(infilename):
    img = imr(infilename)
    data = np.asarray(img, dtype="int32" )

    return data

# function to interpolate new array with a specified spline fit degree
def recspl(inp,m,n):
    x0 = np.arange(1,np.shape(inp)[0]+1,1)
    y0 = np.arange(1,np.shape(inp)[1]+1,1)

    xn = np.arange(1,m,1)
    yn = np.arange(1,n,1)

    xnn, ynn = np.meshgrid(xn, yn)
    f = sp.interpolate.RectBivariateSpline(y0,x0,inp.T)
    zn = f(xn,yn).T

    return xnn,ynn,zn

#%% Run functions

# filename(s)
inp = ask()

# original data
data = load_image(inp)

m = np.shape(data)[0]
n = np.shape(data)[1]

x = np.arange(1,m+1,1)
y = np.arange(1,n+1,1)
Xo, Yo = np.meshgrid(x,y)
Zo = data[:,:,0].T

# interpolated data
xs = 600
ys = 600
Xi, Yi, Zi = recspl(Zo,xs,ys)

#%% Plot

# original
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xo,Yo,Zo)

# interpolated
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xi,Yi,Zi)
