# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:49:57 2018

This file creates a 3D array of values with a Gaussian distribution.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy.ndimage as sp

#%% Create array

# edge length
el = 64

# factor for elongation in Z direction
fz = 1

# scale factor for number of elements
fn = 1

# number of elements in X direction
nx = 12

# number of elements in Y direction
ny = 12

# number of elements in Z direction
nz = 16

# initialize coordinates of element centroids
x = np.linspace(0,el,fn*nx)
y = np.linspace(0,el,fn*ny)
z = np.linspace(0,fz*el,fn*nz)
X,Y,Z = np.meshgrid(x,y,z)

# mean and standard deviation / FWHM for Gaussian Distribution
mean = 1

# option 1
sig = 0.10
fwhm = 2*np.sqrt(2*np.log(2))*sig

# option 2
fwhm = 0.10
sig = fwhm/(2*np.sqrt(2*np.log(2)))

G = np.random.normal(mean,sig,(fn*nx,fn*ny,fn*nz))

# flatten 3D arrays to 1D vectors
xf = X.flatten()
yf = Y.flatten()
zf = Z.flatten()
gf = G.flatten()

# Smooth array using Gaussian filter

# standard deviation for filter in X direction
sx = 1

# standard deviation for filter in Y direction
sy = sx

# standard deviation for filter in Z direction
sz = sx

GS = sp.gaussian_filter(G,(sx,sy,sz))
gsf = GS.flatten()

# create and sort output array (by X, then by Y for ParaView)
out = np.c_[xf,yf,zf,gf,gsf]
inds = np.lexsort([out[:,1],out[:,2]])
output = out[inds]

# export data
head = 'x,y,z,g,gs'
np.savetxt(r'R:\166_320\Transfer\Goumnerov\zWiMi\gauss1.csv', output, delimiter=',', fmt='%10.6f', header=head, comments='')
