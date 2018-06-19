# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:35:08 2017

@author: goumnero
"""
#%% Import modules

import numpy as np
import pandas as pd
from scipy import ndimage
import numdifftools as nd
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%% Import temperature

# Load file
data = pd.read_csv('test.csv',header=8)

# Extract coordinates and temperature
data = data.values
#data = data[data[:,0].argsort()]
#data = data[:np.nonzero(data[:,0]>=400)[0][0],:]
xin = data[:,0]
yin = data[:,1]
zin = data[:,2]
Tin = data[:,3]
xinu = np.unique(xin)
yinu = np.unique(yin)
zinu = np.unique(zin)
nxin = len(xinu)
nyin = len(yinu)
nzin = len(zinu)

yin3, xin3, zin3 = np.meshgrid(yinu,xinu,zinu)

Tint = Tin.reshape((nxin,nyin,nzin),order='F').transpose()

lx = max(xin)
ly = max(yin)
lz = max(zin)

t3d = Tint

check = np.c_[xin3.ravel(),yin3.ravel(),zin3.ravel(),t3d.ravel()]

## Interpolate onto regular grid
#itp = RegularGridInterpolator((yin3, xin3, zin3), Tint, method='nearest')
#nn = 50
#xn = np.linspace(0,lx,nn)
#yn = np.linspace(0,ly,nn)
#zn = np.linspace(0,lz,nn)
#grid = np.ix_(yn, xn, zn)
#t3d_int = itp(grid)

#fig = plt.figure(1)
#ax = fig.add_subplot(111, projection='3d')
#cmm = plt.get_cmap('jet')
#cNorm = matplotlib.colors.Normalize(vmin=np.min(Tint), vmax=np.max(Tint))
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmm)
#ax.scatter(xin,yin,zin, c=scalarMap.to_rgba(Tin), alpha=1)
#scalarMap.set_array(Tin)
#fig.colorbar(scalarMap)
#ax.set_xlim([np.min(xin),np.max(xin)])
#ax.set_ylim([np.min(yin),np.max(yin)])
#ax.set_zlim([np.min(zin),np.max(zin)])
#ax.set_xlabel('X [mm]')
#ax.set_ylabel('Y [mm]')
#ax.set_zlabel('Z [mm]')
#plt.show()

'''
# Function 
def T3D(nx,ny,nz):
    
#    global x
#    
#    out = np.ones([ny,nx,nz])
#    
#    for i in range(nz):
#        out[:,:,i] = np.tile(1300*np.exp(-x/300),(ny,1))    
#    
#    global X,Y,Z
#    
#    out = 293.15 + 0*X + 200*Y + 0*Z
#    
#    return out
    
    unsm = 293.15 + 1700*np.random.rand(nx,ny,nz)
    sig = 3
    out = ndimage.filters.gaussian_filter(unsm, [sig,sig,sig], mode='constant')
    
    return out

#t3d = T3D(nx,ny,nz)
'''

#%% Input

## Size of domain
#lx = 100
#ly = 40
#lz = 40

step = 50 #int(lx/4)
nx = step
ny = step
nz = step

# X, Y, Z, and N
x = xinu #np.linspace(0,lx,nx)
y = yinu #np.linspace(0,ly,ny)
z = zinu #np.linspace(0,lz,nz)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
X,Y,Z = np.meshgrid(x,y,z)
r = np.array([x,y,z], dtype=float).T
#r = np.array([xinu,yinu,zinu], dtype=float).T

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
#xo = X.ravel()
#yo = Y.ravel()
#zo = Z.ravel()
xo = xin3.ravel()
yo = yin3.ravel()
zo = zin3.ravel()

to = t3d.ravel()
no = n3d.ravel()

fo = np.c_[xo,yo,zo,to,no]
foo = fo[fo[:,1].argsort()]

npp = 100
xp = foo[::npp,0]
yp = foo[::npp,1]
zp = foo[::npp,2]
npl = foo[::npp,3]
cs = npl

#x2d = X[:,:,0]
#y2d = Y[:,:,0]
#z2d = Z[:,:,0]
    
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
    ax.set_xlim([np.min(xinu),np.max(xinu)])
    ax.set_ylim([np.min(yinu),np.max(yinu)])
    ax.set_zlim([np.min(zinu),np.max(zinu)])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.show()

#%% Export output

#fo = np.float64(sorted(fo,key=lambda x: (x[2],x[1])))
#fo = fo[fo[:,2].argsort()]

# Ray trajectory
#xmax = np.argmin(np.abs(rx-lx))
#xout = np.arange(0,lx,0.01)
#yout = np.interp(xout,rx,ry)
#zout = np.interp(xout,rx,rz)
#rto = np.c_[xout,yout,zout]
rto = np.c_[rx,ry,rz]

xc = rto[rto[:,0]<=lx,:]
yc = xc[xc[:,1]<=ly,:]
zc = yc[yc[:,2]<=lz,:]

rout = zc

rx = rout[:,0]
ry = rout[:,1]
rz = rout[:,2]

# Export
head1 = 'TITLE = "3D N and T"\nVARIABLES = "X","Y","Z","T","N"\nZONE T = "T and N", I=%d, J=%d, K=%d' % (ny,nx,nz)
np.savetxt('3D_TN.dat',fo,fmt=['%8.4f','%8.4f','%8.4f','%6.2f','%10.6f'],delimiter='\t',header=head1,comments='')

#head2 = 'TITLE = "3D XYZ"\nVARIABLES = "X","Y","Z"\nZONE T = "Frame 1", I=%d, J=%d' % (len(xout),3)
#np.savetxt('3D_XYZ.dat',rto,fmt=['%6.4e','%6.10e','%6.10e'],delimiter='\t',header=head2,comments='')
head2 = 'X,Y,Z'
np.savetxt('3D_XYZ.csv',rout,fmt=['%8.4f','%8.4f','%8.4f'],delimiter=',',header=head2,comments='')
