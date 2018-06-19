# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:35:08 2017

@author: goumnero
"""
#%% Import modules

import numpy as np
import scipy as sp
import numdifftools as nd
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%% Input

# Starting point
r_0 = [0,10]

# Starting angle
theta = 10

# Size of domain
lx = 40
ly = 40

# Integration range
t_range = np.arange(0,lx+5,0.01)

#%% Load data

step = 50
nx = step
ny = step

# X, Y, and N
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X,Y = np.meshgrid(x,y)
r = np.array([x,y]).T

#N = 1+np.random.rand(nx,ny)

#%% Refractive Index function

### 2D ###

def N2D(r):
    
    if len(r) > 2:
        x,y = np.meshgrid(r[:,0],r[:,1])
    else:
        x = r[0]
        y = r[1]
      
        
#    out = 1.00+0.000*x+0.3*y
#    out = np.sqrt(2) - (np.sqrt(2)-1)/(1+np.exp(-(x - 2)/0.03))
    out = 1+0.01*x*np.sin(x)
    
    return out
        
#    id_x = np.argmin(np.abs(x-xx[0,:]))        # Index of closest x element
#    id_y = np.argmin(np.abs(y-yy[:,0]))        # Index of closest y element
#    
#    return N[id_y,id_x]

n2d = N2D(r)

#%% Set Initial Conditions

########################## Starting point #####################################

### 2D ###

id_x = np.argmin(np.abs(x-r_0[0]))        # Index of closest x element
id_y = np.argmin(np.abs(y-r_0[1]))        # Index of closest y element
n_0 = N2D(r_0) #N[id_y,id_x]

######################### Initial incident angle ##############################

theta_0 = theta*(np.pi/180)

############ Initial velocity based on parametrization constraint #############
'''
n(r)/mag(r.) = 1, n(r) = mag(r.) = sqrt((dx/dt)^2+(dy/dt)^2)
dx/dt = n(r_0)*cos(theta_0)
dy/dt = n(r_0)*sin(theta_0)
'''

### 2D ###

# Derivative
dxdt = np.float64(n_0*np.cos(theta_0))
dydt = np.float64(n_0*np.sin(theta_0))
#
## Initial velocity
v_0 = [dxdt,dydt]

#%% Define functions and solve ray path equation

### 2D ###

# Gradient
n2dgx, n2dgy = np.gradient(n2d,dx,dy,axis=(1,0))
grd_n2d = nd.Gradient(N2D)

# Compute the differential
def diff_y2d(y, t):

    xx = y[0]
    yy = y[1]
    rr = [xx,yy]
    
    n_t = N2D(rr)                           # starting RI
    grd = grd_n2d([xx,yy])                  # gradient
    
    return [y[2], y[3], grd[0]*n_t, grd[1]*n_t]

# Integration
sol2d = odeint(diff_y2d,r_0 + v_0,t_range)

#%% Plot

rx = sol2d[:,0]
ry = sol2d[:,1]

rto = np.c_[rx,ry]
xc = rto[rto[:,0]<=lx,:]
yc = xc[xc[:,1]<=ly,:]

rout = yc

rx = rout[:,0]
ry = rout[:,1]

plt.figure(1)
plt.plot(rx, ry, 'y',linewidth=2)
pcm = plt.pcolormesh(X,Y,n2d,cmap = 'Greys',vmin=1,vmax=np.max(n2d))
cbar = plt.colorbar(pcm)
cbar.ax.set_ylabel("$n$  Refractive index")
plt.plot(r_0[0],r_0[1],'ro')
plt.quiver(r_0[0],r_0[1],v_0[0],v_0[1])
plt.xlim([min(x),max(x)])
plt.ylim([min(y),max(y)])
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.legend(['dY: %4.2f mm' % (ry[-1]-ry[0])])
plt.show()

print('\nDeflection: %4.2f mm' % (ry[-1]-ry[0]))
