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

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%% Input

# Starting point
r_0 = [0,7.5,7.5]

# Domain size
lx = 400
ly = 15
lz = 15

# Starting angle
theta = 0

# Integration range
t_range = np.arange(0,lx+5,0.01)
#%% Load data

step = 20
nx = 2*step
ny = step
nz = step

# X, Y, and N
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)
z = np.linspace(0,lz,nz)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
X,Y,Z = np.meshgrid(x,y,z)
r = np.array([x,y,z]).T

#%% Refractive Index function

### 3D ###

def N3D(pos):
    
    if type(pos) == np.ndarray:
        x,y,z = np.meshgrid(pos[0],pos[1],pos[2])
        
        out = np.zeros(np.shape(x))
        ox = np.shape(x)[1]
        oy = np.shape(x)[0]
        oz = np.shape(x)[2]
        
        for i in range(ox):
            for j in range(oy):
                for k in range(oz):
                    out[j,i,k] = 1+0.0*x[j,i,k]+0.003*y[j,i,k]+0.001*z[j,i,k]

    else:
        x = pos[0]
        y = pos[1]
        z = pos[2]
    
        out = 1+0*x+0.003*y+0.001*z

    return out

#def N3D(xi,yi,zi):
#    
#    if np.size(xi) > 1:
#        x,y,z = np.meshgrid(xi,yi,zi)
#    else:
#        x = xi
#        y = yi
#        z = zi
#    
#    return 1+0.0*x+0.003*y+0.001*z

#%% Set Initial Conditions

########################## Starting point #####################################

### 3D ###

#id_x = np.argmin(np.abs(x-r_0[0]))        # Index of closest x element
#id_y = np.argmin(np.abs(y-r_0[1]))        # Index of closest y element
#id_z = np.argmin(np.abs(z-r_0[2]))        # Index of closest z element
n_0 = N3D(r_0)

######################### Initial incident angle ##############################

theta_0 = theta*(np.pi/180)

############ Initial velocity based on parametrization constraint #############
'''
n(r)/mag(r.) = 1, n(r) = mag(r.) = sqrt((dx/dt)^2+(dy/dt)^2+(dz/dt)^2)
dx/dt = n(r_0)*cos(theta_0)
dy/dt = n(r_0)*sin(theta_0)
dz/dt = n(r_0)*cos(theta_0)
'''
### 3D ###

a = 0.000
dxdt = np.float64(n_0*np.cos(theta_0)-a)
dydt = np.float64(n_0*np.sin(theta_0)-a)
dzdt = np.float64(np.sqrt(n_0**2-dxdt**2-dydt**2))
ch = np.float64(np.sqrt(dxdt**2+dydt**2+dzdt**2))/n_0

v_0 = [dxdt,dydt,dzdt]
#v_0 = [n_0,0,0]

#%% Define functions and solve ray path equation

### 3D ###

n3d = N3D(r)

# Gradient
n3dgx, n3dgy, n3dgz = np.gradient(n3d,dx,dy,dz,axis=(1,0,2))
grd_n3d = nd.Gradient(N3D)

# Compute the differential
def diff_y3d(y, t):
    
    xx = y[0]
    yy = y[1]
    zz = y[2]
    rr = [xx,yy,zz]
    
#    print(xx)
#    print(yy)
#    print(zz)
#    
    n_t = N3D(rr)                                   # starting RI    
    grd = grd_n3d(rr)                               # gradient
 
    return [y[3], y[4], y[5], grd[0]*n_t, grd[1]*n_t, grd[2]*n_t]

# Integration
sol3d = odeint(diff_y3d,r_0 + v_0,t_range)

#%% Plot

# 3D field output
xo = X.ravel()
yo = Y.ravel()
zo = Z.ravel()
no = n3d.ravel()
fo = np.c_[xo,yo,zo,no]
fo = fo[fo[:,0].argsort()]

npp = 2000
xp = fo[::npp,0]
yp = fo[::npp,1]
zp = fo[::npp,2]
npl = fo[::npp,3]
cs = npl

x2d = X[:,:,0]
y2d = Y[:,:,0]
z2d = Z[:,:,0]
    
rx = sol3d[:,0]
ry = sol3d[:,1]
rz = sol3d[:,2]

rto = np.c_[rx,ry,rz]

xc = rto[rto[:,0]<=lx,:]
yc = xc[xc[:,1]<=ly,:]
zc = yc[yc[:,2]<=lz,:]

rout = zc

rx = rout[:,0]
ry = rout[:,1]
rz = rout[:,2]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(rx, ry, rz, 'y',linewidth=2)
ax.quiver(r_0[0],r_0[1],r_0[2],v_0[0],v_0[1],v_0[2])
ax.scatter(r_0[0],r_0[1],r_0[2],color='r')
cmm = plt.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmm)
ax.scatter(xp,yp,zp, c=scalarMap.to_rgba(cs), alpha=0.3)
scalarMap.set_array(cs)
fig.colorbar(scalarMap)
ax.set_xlim([min(x),max(x)])
ax.set_ylim([0,max(y)])
ax.set_zlim([0,max(z)])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.show()
