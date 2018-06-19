# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:17:28 2018

This file reads in an image and computes relevant roughness parameters.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy as sp
import scipy.ndimage as nd

# import matplotlib for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%% Import image

#img = sp.misc.imread('1.jpg')

# import image
img0 = nd.imread('1.jpg', mode='L')

# crop image
imgbw = img0[:1300,:1800]

# set calibration scales
sxy = (100e-6)/142.29
sz = 0.1e-6

# function to compute roughness parameters
def georough(inp):
    
    xs, ys = np.shape(inp)
    
    x = np.arange(0,xs)*sxy
    y = np.arange(0,ys)*sxy
    z = (inp.T)*sz
    
    X,Y = np.meshgrid(x,y)
    Z = z
    Z = Z - np.mean(Z)
    xf = X.flatten()
    yf = Y.flatten()
    zf = Z.flatten()
    
    Ra = np.sum(abs(Z))/len(zf)
    Rv = np.min(Z)
    Rp = np.max(Z)
    Rt = Rp - Rv
    #Rstd = np.std(Z)
    Rq = np.sqrt(np.mean(Z**2))
    Rsk = (1/(len(zf)*Rq**3))*np.sum(Z**3)
    Rku = (1/(len(zf)*Rq**4))*np.sum(Z**4)

    return X,Y,Z,xf,yf,zf,Ra,Rv,Rp,Rt,Rq,Rsk,Rku

# run multiple windows of image for calculation
n = 10

out = np.zeros((n-1,5))

for i in range(1,n):
    
    inp = imgbw[:int(1300/i),:int(1800/i)]

    X,Y,Z,xf,yf,zf,Ra,Rv,Rp,Rt,Rq,Rsk,Rku = georough(inp)
    
    out[i-1,0] = Ra
    out[i-1,1] = Rt
    out[i-1,2] = Rq
    out[i-1,3] = Rsk
    out[i-1,4] = Rku

# save output
avgout = np.mean(out, axis=0).T
stdout = np.std(out, axis=0).T
out = np.c_[out.T,avgout,stdout]
out = out.T

#%% Plot

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.scatter(xf,yf,zf)
#
#print('\nRa = %4.2f um' % (Ra*1e6))
