# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:35:08 2017

@author: goumnero
"""
#%% Import modules

import numpy as np
from scipy import ndimage
import numdifftools as nd
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

#%% Input

# Starting point
r_0 = [0,200]

# Starting angle
theta = 0

# Size of domain
lx = 400
ly = 400

ll = np.sqrt(lx**2+ly**2)

# Integration range
t_range = np.arange(0,np.max([lx,ly,ll])+1,1)

#%% Load data

step = 50 #int(lx/40)
nx = step
ny = step

# X, Y, and N
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X,Y = np.meshgrid(x,y)
r = np.array([x,y], dtype=float).T

#%% Refractive Index function

def T2D(nx,ny):
    
#    global x
#    
#    tx = 1300*np.exp(-x/300)
#
#    return np.tile(tx,(ny,1))

    unsm = 293.15 + 1200*np.random.rand(nx,ny)
    
    sig = 1
    return ndimage.filters.gaussian_filter(unsm, [sig,sig], mode='constant')

t2d = T2D(nx,ny)

nn = 10
xn = np.arange(0,lx,nn)
yn = np.arange(0,ly,nn)
itp = RegularGridInterpolator((y, x), t2d, method='nearest')

grid = np.ix_(yn, xn)
t2d_int = itp(grid)

def N2D(r):
    
    if len(r) > 2:
        
        return 1+(0.000293*293.15)/T2D(nx,ny)
    
    else:
        
        global x,y
        
        xx = r[0]
        yy = r[1]
        
        idx = np.argmin(np.abs(x-xx))        # Index of closest x element
        idy = np.argmin(np.abs(y-yy))        # Index of closest y element
        
        return 1+(0.000293*293.15)/t2d[idy,idx]

n2d = N2D(r)

#%% Set Initial Conditions

########################## Starting point #####################################

id_x = np.argmin(np.abs(x-r_0[0]))        # Index of closest x element
id_y = np.argmin(np.abs(y-r_0[1]))        # Index of closest y element
n_0 = N2D(r_0)

######################### Initial incident angle ##############################

theta_0 = theta*(np.pi/180)

############ Initial velocity based on parametrization constraint #############
'''
n(r)/mag(r.) = 1, n(r) = mag(r.) = sqrt((dx/dt)^2+(dy/dt)^2)
dx/dt = n(r_0)*cos(theta_0)
dy/dt = n(r_0)*sin(theta_0)
'''

# Derivative
dxdt = np.float64(n_0*np.cos(theta_0))
dydt = np.float64(n_0*np.sin(theta_0))

# Initial velocity
v_0 = [dxdt,dydt]

#%% Define functions and solve ray path equation

# Gradient
n2dgx, n2dgy = np.gradient(n2d,dx,dy,axis=(1,0))
grd_n2d = nd.Gradient(N2D)

# Compute the differential
def diff_y2d(y, t):

    xx = y[0]
    yy = y[1]
    rr = [xx,yy]
    
    n_t = N2D(rr)                         # starting RI
    grd = grd_n2d(rr)                     # gradient
    
    return [y[2], y[3], grd[0]*n_t, grd[1]*n_t]

# Integration
sol2d = odeint(diff_y2d,r_0 + v_0,t_range)

#%% Plot

rx = sol2d[:,0]
ry = sol2d[:,1]
delx = rx[-1]-rx[0]
dely = ry[-1]-ry[0]

f, (ax1, ax2) = plt.subplots(1,2)
pcm1 = ax1.pcolormesh(X,Y,t2d,cmap = 'jet')#,vmin=np.mean(t2d)-100,vmax=np.mean(t2d)+100)
cbar1 = plt.colorbar(pcm1, ax=ax1)
cbar1.ax.set_ylabel("$T$  Temperature")
ax1.plot(rx, ry, 'y',linewidth=2)
ax1.plot(r_0[0],r_0[1],'ro')
ax1.quiver(r_0[0],r_0[1],v_0[0],v_0[1])
ax1.set_xlim([min(x),max(x)])
ax1.set_ylim([min(y),max(y)])
ax1.set_xlabel('X [mm]')
ax1.set_ylabel('Y [mm]')

pcm2 = ax2.pcolormesh(X,Y,n2d,cmap = 'Greys')#,vmin=1,vmax=1.001*np.mean(n2d))
cbar2 = plt.colorbar(pcm2, ax=ax2)
cbar2.ax.set_ylabel("$n$  Refractive index")
ax2.plot(rx, ry, 'y',linewidth=2)
ax2.plot(r_0[0],r_0[1],'ro')
ax2.quiver(r_0[0],r_0[1],v_0[0],v_0[1],color='r')
ax2.set_xlim([min(x),max(x)])
ax2.set_ylim([min(y),max(y)])
ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.annotate('dx: %4.2f mm\ndy: %4.2f mm' % (delx,dely), xy=(0.95, 0.95), xycoords='axes fraction', size=12, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
plt.show()

#fig = plt.figure(2)
#ax1 = fig.add_subplot(111)
#l1 = ax1.plot(rx,ry-r_0[1],'r',label='Y-yo')
#l2 = ax1.plot(rx,np.gradient(ry,dx),'k',label='dy/dx')
#ax2 = ax1.twinx()
#l3 = ax2.plot(rx,np.arctan(np.gradient(ry,dx))*180/np.pi,'g',label='Theta')
#lns = l1+l2+l3
#labs = [l.get_label() for l in lns]
#ax1.legend(lns,labs,loc=4)
#ax1.grid()
#ax1.set_xlim([min(x),max(x)])
#ax1.set_ylim([-max(y)/2,max(y)/2])
#ax1.set_xlabel('X [mm]')
#ax1.set_ylabel('Y [mm]')
#ax2.set_ylabel('Angle [deg]')
#plt.show()

#plt.figure(3)
#plt.hist(n2d,bins=5)

#print('\nDeflection: delX = %4.4f mm, delY = %4.4f mm' % (delx,dely))

#rxc = rx[1]-rx[0]
#ryc = ry[1]-ry[0]
#thetac = np.arctan2(ryc,rxc)*180/np.pi
#drx = v_0[0]-rxc
#dry = v_0[1]-ryc
#dtheta = np.arctan2(v_0[1],v_0[0])*180/np.pi-thetac
#
#print('\nVelocity:  vx = %4.20f mm, vy = %4.20f mm, theta = %4.20f deg' % (v_0[0],v_0[1],np.arctan(v_0[1]/v_0[0])*180/np.pi))
#print('Vector(100):   rx = %4.20f mm, ry = %4.20f mm, theta = %4.20f deg' % (rxc,ryc,thetac))
#print('\nDifference dtheta = %4.4e deg'% dtheta)
#%% Export output

# 2D field output
xo = X.ravel()
yo = Y.ravel()

to = t2d.ravel()
no = n2d.ravel()

fo = np.c_[xo,yo,to,no]

# Ray trajectory
#xmax = np.argmin(np.abs(rx-lx))
#xout = np.arange(0,lx,0.01)
#yout = np.interp(xout,rx,ry)
#rto = np.c_[xout,yout]
rto = np.c_[rx,ry]

xc = rto[rto[:,0]<=lx,:]
yc = xc[xc[:,1]<=ly,:]

rout = yc

# Export
head1 = 'TITLE = "2D N and T"\nVARIABLES = "X","Y","T","N"\nZONE T="Frame 1", I=%d, J=%d' % (ny,nx)
np.savetxt('2D_TN.dat',fo,fmt=['%8.4f','%8.4f','%6.2f','%10.6f'],delimiter='\t',header=head1,comments='')
#
#head2 = 'TITLE = "2D XYZ"\nVARIABLES = "X","Y"\nZONE T="Frame 1", I=%d, J=%d' % (len(xout),2)
#np.savetxt('2D_XY.dat',rto,fmt=['%6.4e','%6.10e'],delimiter='\t',header=head2,comments='')
head2 = 'X,Y'
np.savetxt('2D_XY.csv',rout,fmt=['%8.4f','%8.4f'],delimiter=',',header=head2,comments='')
