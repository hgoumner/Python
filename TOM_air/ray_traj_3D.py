# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:35:08 2017

@author: goumnero
"""
#%% Import modules

import numpy as np
import pandas as pd
import numdifftools as nd
from scipy.integrate import odeint

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% Import temperature

# Load file
data = pd.read_csv('test.csv',header=8)

# Extract coordinates and temperature
data = data.values
xin = data[:,0]
yin = data[:,1]
zin = data[:,2]
Tin = data[:,3]
x = np.unique(xin)
y = np.unique(yin)
z = np.unique(zin)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
nx = len(x)
ny = len(y)
nz = len(z)
lx = int(max(x)-min(x))
ly = int(max(y)-min(y))
lz = int(max(z)-min(z))

r = np.array([x,y,z], dtype=float).T

y3, x3, z3 = np.meshgrid(y,x,z)

Tint = Tin.reshape((ny,nx,nz),order='F').transpose()

t3d = Tint

check = np.c_[x3.ravel(),y3.ravel(),z3.ravel(),t3d.ravel()]
inds = np.lexsort((check[:,1],check[:,0]))
check = check[inds] 

#fig = plt.figure(1)
#ax = fig.add_subplot(111, projection='3d')
#cmm = plt.get_cmap('jet')
#cNorm = matplotlib.colors.Normalize(vmin=np.min(Tint), vmax=np.max(Tint))
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmm)
#nnn = 10
#ax.scatter(xin[::nnn],yin[::nnn],zin[::nnn], c=scalarMap.to_rgba(Tin[::nnn]), alpha=1)
#scalarMap.set_array(Tin)
#fig.colorbar(scalarMap)
#ax.set_xlim([np.min(xin),np.max(xin)])
#ax.set_ylim([np.min(yin),np.max(yin)])
#ax.set_zlim([np.min(zin),np.max(zin)])
#ax.set_xlabel('X [mm]')
#ax.set_ylabel('Y [mm]')
#ax.set_zlabel('Z [mm]')
#plt.show()

#%% Input

# Starting point
r_0 = [0,0,np.mean(z)]

# Starting angle
theta = 0

#%% Refractive Index function

def N3D(r):
    
    if len(r) > 3:
        
        return 1+(0.000293*293.15)/t3d
    
    else:
        
        global x,y,z
        
        xx = r[0]
        yy = r[1]
        zz = r[2]
        
        idx = np.argmin(np.abs(x-xx))        # Index of closest x element
        idy = np.argmin(np.abs(y-yy))        # Index of closest y element
        idz = np.argmin(np.abs(z-zz))        # Index of closest z element
        
        return 1+(0.000293*293.15)/t3d[idy,idx,idz]

n3d = N3D(r)

#%% Set Initial Conditions

########################## Starting point #####################################

id_x = np.argmin(np.abs(x-r_0[0]))        # Index of closest x element
id_y = np.argmin(np.abs(y-r_0[1]))        # Index of closest y element
id_z = np.argmin(np.abs(z-r_0[2]))        # Index of closest z element
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

# Derivative
a = 0.000
dxdt = np.float64(n_0*np.cos(theta_0)-a)
dydt = np.float64(n_0*np.sin(theta_0)-a)
dzdt = 0 #np.float64(np.sqrt(n_0**2-dxdt**2-dydt**2))
ch = np.float64(np.sqrt(dxdt**2+dydt**2+dzdt**2))/n_0

# Initial velocity
v_0 = [dxdt,dydt,dzdt]

#%% Define functions and solve ray path equation

# Gradient
n3dgx, n3dgy, n3dgz = np.gradient(n3d,dx,dy,dz,axis=(1,0,2))
grd_n3d = nd.Gradient(N3D)

# Compute the differential
def diff_y3d(y, t):
    
    xx = y[0]
    yy = y[1]
    zz = y[2]
    rr = [xx,yy,zz]
    
    n_t = N3D(rr)                            # starting RI    
    grd = grd_n3d(rr)                        # gradient

    return [y[3], y[4], y[5], grd[0]*n_t, grd[1]*n_t, grd[2]*n_t]

# Integration
t_range = np.arange(0,np.max([lx,ly,lz])+5,0.01)
sol3d = odeint(diff_y3d,r_0 + v_0,t_range)

#%% Plot

# 3D field output
xo = x3.ravel()
yo = y3.ravel()
zo = z3.ravel()
to = t3d.ravel()
no = n3d.ravel()

fo = np.c_[xo,yo,zo,to,no]
foo = fo[fo[:,0].argsort()]

npp = 100
xp = foo[::npp,0]
yp = foo[::npp,1]
zp = foo[::npp,2]
npl = foo[::npp,3]
cs = npl
    
rx = sol3d[:,0]
ry = sol3d[:,1]
rz = sol3d[:,2]

delx = rx[-1]-rx[0]
dely = ry[-1]-ry[0]
delz = rz[-1]-rz[0]

pltt = 0

if pltt == 1:
    
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
    ax.set_xlim([np.min(x),np.max(x)])
    ax.set_ylim([np.min(y),np.max(y)])
    ax.set_zlim([np.min(z),np.max(z)])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.show()

#%% Export output

# Ray trajectory
rto = np.c_[rx,ry,rz]
xc = rto[rto[:,0]>=min(x),:]
xc = xc[xc[:,0]<=max(x),:]
yc = xc[xc[:,1]>=min(y),:]
yc = yc[yc[:,1]<=max(y),:]
zc = yc[yc[:,2]>=min(z),:]
zc = zc[zc[:,2]<=max(z),:]
rout = zc

# Export
head1 = 'TITLE = "3D N and T"\nVARIABLES = "X","Y","Z","T","N"\nZONE T = "T and N", I=%d, J=%d, K=%d' % (ny,nx,nz)
np.savetxt('3D_TN.dat',fo,fmt=['%8.4f','%8.4f','%8.4f','%6.2f','%10.6f'],delimiter='\t',header=head1,comments='')

#head2 = 'TITLE = "3D XYZ"\nVARIABLES = "X","Y","Z"\nZONE T = "Frame 1", I=%d, J=%d' % (len(xout),3)
#np.savetxt('3D_XYZ.dat',rto,fmt=['%6.4e','%6.10e','%6.10e'],delimiter='\t',header=head2,comments='')
head2 = 'X,Y,Z'
np.savetxt('3D_XYZ.csv',rout,fmt=['%8.4f','%8.4f','%8.4f'],delimiter=',',header=head2,comments='')

print('\ndY: %10.8e m, dZ: %10.8e m' % (dely/1000,delz/1000))
