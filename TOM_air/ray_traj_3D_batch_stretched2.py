# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:35:08 2017

@author: goumnero
"""
#%% Import modules

import numpy as np
import pandas as pd
import scipy.interpolate as si
import numdifftools as nd
from scipy.integrate import odeint

#%% Input

# Load file
direc = '1 K/KFH/T15_5055'
data = pd.read_csv(direc + '/1K_T15_5055.csv',header=8)

#%% Import temperature, compute refractive index, export data

# Extract coordinates and temperature
def get_temp(data):
    
    data = data.values                                                          # Convert data to array
    xin = data[:,0]+10                                                          # Get X coordinates
    yin = data[:,1]                                                             # Get Y coordinates
    zin = data[:,2]                                                             # Get Z coordinates
    Tin = data[:,3]                                                             # Get Temperature
    x = np.unique(xin)                                                          # Get unique X coordinates
    y = np.unique(yin)                                                          # Get unique Y coordinates
    z = np.unique(zin)                                                          # Get unique Z coordinates                                                            # Center Z coordinates to optical axis
    dx = x[1] - x[0]                                                            # Get X step
    dy = y[1] - y[0]                                                            # Get Y step
    dz = z[1] - z[0]                                                            # Get Z step
    nx = len(x)                                                                 # Get number of X coordinates
    ny = len(y)                                                                 # Get number of Y coordinates 
    nz = len(z)                                                                 # Get number of Z coordinates
    lx = int(max(x)-min(x))                                                     # Get range of X
    ly = int(max(y)-min(y))                                                     # Get range of Y
    lz = int(max(z)-min(z))                                                     # Get range of Z
    
    r = np.array([x,y,z]).T                                                     # Create position array
    y3, x3, z3 = np.meshgrid(y,x,z)                                             # Create 3D coordinate array
    
    t = Tin.reshape((ny,nx,nz),order='F').transpose()                         # Reshape temperature input to 3D array
    
    '''
    Check if reshaped data matches input
    check = np.c_[x3.ravel(),y3.ravel(),z3.ravel(),t3d.ravel()]
    inds = np.lexsort((check[:,1],check[:,0]))
    check = check[inds]
    '''
    
    return x, y, z, dx, dy, dz, nx, ny, nz, lx, ly, lz, r, x3, y3, z3, t      # Output parameters

# Refractive Index function according to Gladstone-Dale Law

def N3D(pos):
    
    global t
    
    beta = 0.000293
    Ts = 293.15
    
    # Compute whole 3D array or determine starting point
    if type(pos) == np.ndarray:
        
        xg,yg,zg = np.meshgrid(pos[0],pos[1],pos[2])
        out = np.zeros(np.shape(xg))
        ox = np.shape(xg)[1]
        oy = np.shape(xg)[0]
        oz = np.shape(xg)[2]
        
        for i in range(ox):
            for j in range(oy):
                for k in range(oz):
                    out[j,i,k] = 1 + beta*Ts/t[j,i,k]
        
#        out = 1 + beta*Ts/t
    
    else:
        
        global x,y,z
        
        xx = pos[0]
        yy = pos[1]
        zz = pos[2]
        
        idx = np.argmin(np.abs(x-xx))                                          # Index of closest x element
        idy = np.argmin(np.abs(y-yy))                                          # Index of closest y element
        idz = np.argmin(np.abs(z-zz))                                          # Index of closest z element
        out = 1 + beta*Ts/t[idy,idx,idz]
        
    return out                                                                  # Output parameters
    
#def N3D(r):
#    
#    if len(r) > 3:
#        yyy,xxx,zzz = np.meshgrid(r[:,1],r[:,0],r[:,2])
#    else:
#        xxx = r[0]
#        yyy = r[1]
#        zzz = r[2]
#    
#    return 1+0.0*xxx+0.000*yyy+0.02*zzz

def tinterp(x,y,z,t,p):
    
    # Interpolation
    xx = x #np.linspace(np.min(x3),np.max(x3),p)
    yy = np.linspace(np.min(y),np.max(y),p)
    zz = np.linspace(np.min(z),np.max(z),p)
      
    dx = xx[1] - xx[0]                                                            # Get X step
    dy = yy[1] - yy[0]                                                            # Get Y step
    dz = zz[1] - zz[0]                                                            # Get Z step
    nx = len(xx)                                                                 # Get number of X coordinates
    ny = len(yy)                                                                 # Get number of Y coordinates 
    nz = len(zz)                                                                 # Get number of Z coordinates
    
    y3i, x3i, z3i = np.meshgrid(yy,xx,zz)
    r = np.array([xx,yy,zz]).T                                                     # Create position array

#    itp = si.RegularGridInterpolator((y, x, z), t, method='nearest')
#    grid = np.ix_(yy, xx, zz)
#    ti = itp(grid)  
    
    ti = si.interpn((y,x,z), t, np.array([y3i,x3i,z3i]).T)
    ti = ti.swapaxes(1,2)
    
    return x3i, y3i, z3i, dx, dy, dz, nx, ny, nz, ti, r

# Get data
x, y, z, dx, dy, dz, nx, ny, nz, lx, ly, lz, r, x3, y3, z3, t = get_temp(data)
offset = 0 #np.mean(z)

# Interpolate onto finer grid
p = 8
x3i, y3i, z3i, dxi, dyi, dzi, nxi, nyi, nzi, ti, ri = tinterp(x,y,z,t,p+1)
x3 = x3i
y3 = y3i
z3 = z3i
dx = dxi
dy = dyi
dz = dzi
nx = nxi
ny = nyi
nz = nzi
tin = t
t = ti.T
r = ri

# Refractive Index and gradients
n = N3D(r)
ngy, ngx, ngz = np.gradient(n,dy,dx,dz,axis=(1,0,2))
tgy, tgx, tgz = np.gradient(t,dy,dx,dz,axis=(1,0,2))

# Export 3D field output
xo = x3.ravel()
yo = y3.ravel()
zo = z3.ravel() - offset
to = t.ravel()
no = n.ravel()
tgxf = tgx.ravel()
tgyf = tgy.ravel()
tgzf = tgz.ravel()
ngxf = ngx.ravel()
ngyf = ngy.ravel()
ngzf = ngz.ravel()
ngtot = np.sqrt(ngxf**2+ngyf**2+ngzf**2)
ngxp = abs(ngxf/ngtot)
ngyp = abs(ngyf/ngtot)
ngzp = abs(ngzf/ngtot)
fo = np.c_[xo,yo,zo,to,tgxf,tgyf,tgzf,no,ngxf,ngyf,ngzf,ngtot,ngxp,ngyp,ngzp]
fo = np.float64(sorted(fo,key=lambda x: (x[2],x[1])))
head1 = 'TITLE = "3D N and T"\nVARIABLES = "X","Y","Z","T","Tx","Ty","Tz","N","Nx","Ny","Nz","Ntot","Nxt","Nyt","Nzt"\nZONE T = "T and N", I=%d, J=%d, K=%d' % (nx,ny,nz)
np.savetxt(direc + '/KF_3D_TN.dat',fo,fmt='%10.6e',delimiter='\t',header=head1,comments='')

#%% Compute and export ray trajectory

# Gradient
grd_n3d = nd.Gradient(N3D)

# Integration range
t_range = np.arange(0,np.max([lx,ly,lz])+5,0.01)

def start_point(x,y,z,r_0,theta):
    
    ########################## Starting point #####################################
    
#    id_x = np.argmin(np.abs(x-r_0[0]))        # Index of closest x element
#    id_y = np.argmin(np.abs(y-r_0[1]))        # Index of closest y element
#    id_z = np.argmin(np.abs(z-r_0[2]))        # Index of closest z element
    n_0 = N3D(r_0)
    
    ######################### Initial incident angle ##############################
    
#    theta_0 = theta*(np.pi/180)
    
    ############ Initial velocity based on parametrization constraint #############
    '''
    n(r)/mag(r.) = 1, n(r) = mag(r.) = sqrt((dx/dt)^2+(dy/dt)^2+(dz/dt)^2)
    dx/dt = n(r_0)*cos(theta_0)
    dy/dt = n(r_0)*sin(theta_0)
    dz/dt = n(r_0)*cos(theta_0)
    '''
    
    # Derivative
#    a = 0.000
    dxdt = -n_0 #np.float64(n_0*np.cos(theta_0)-a)
    dydt = 0.0   #np.float64(n_0*np.sin(theta_0)-a)
    dzdt = 0.0   #np.float64(np.sqrt(n_0**2-dxdt**2-dydt**2))
#    ch = np.float64(np.sqrt(dxdt**2+dydt**2+dzdt**2))/n_0
    
    # Initial velocity
    v_0 = [dxdt,dydt,dzdt]
    
    return v_0                                                                  # Output parameter

# Compute the differential
def diff_y3d(y, t):
    
    xx = y[0]
    yy = y[1]
    zz = y[2]
    rr = [xx,yy,zz]
    
#    print(rr)
    
    n_t = N3D(rr)                                                               # starting RI    
    grd = grd_n3d(rr)                                                           # gradient

    return [y[3], y[4], y[5], grd[0]*n_t, grd[1]*n_t, grd[2]*n_t]               # Output parameter

# Export results
def exp_res(sol3d,x,y,z,nx,ny,nz,r_0):
        
    rx = sol3d[:,0]                                                             # Get x coordinate of ray
    ry = sol3d[:,1]                                                             # Get y coordinate of ray
    rz = sol3d[:,2]                                                             # Get z coordinate of ray
    
    delx = rx[-1]-rx[0]                                                         # Get x deflection (last - first point)
    dely = ry[-1]-ry[0]                                                         # Get y deflection (last - first point)
    delz = rz[-1]-rz[0]                                                         # Get z deflection (last - first point)
    
    # Ray trajectory
    rto = np.c_[rx,ry,rz]
    xc = rto[rto[:,0]>=min(x),:]
    xc = xc[xc[:,0]<=max(x),:]
    yc = xc[xc[:,1]>=min(y),:]
    yc = yc[yc[:,1]<=max(y),:]
    zc = yc[yc[:,2]>=min(z),:]
    zc = zc[zc[:,2]<=max(z),:]
    rout = zc
    npp = 100
    rout = np.c_[rout[::npp,0],rout[::npp,1],rout[::npp,2]]
    
#    vals = 8000                                                                 # Set number of values to be interpolated
#    rxint = np.linspace(min(x),round(max(x)),vals+1)                            # Set interpolated x array
#    ryint = np.interp(rxint,rout[:,0],rout[:,1])                                # Set interpolated y array
#    rzint = np.interp(rxint,rout[:,0],rout[:,2])                                # Set interpolated z array
#    rout = np.c_[rxint,ryint,rzint]                                             # Put ray trajectory in array
    
    devx = np.abs(rout[:,0] - rout[0,0])/1000
    devy = (rout[:,1] - ry[0])/1000
    devz = (rout[:,2] - rz[0])/1000
    devtot = np.sqrt(devy**2+devz**2)
    rout = np.c_[rout,devx,devy,devz,devtot]
#    print('\ndY: %10.8e m, dZ: %10.8e m' % (dely/1000,delz/1000))
    
    return rout, delx, dely, delz

# Allocate deflection arrays
xd = []
yd = []
zd = []
delx = []
dely = []
delz = []
rall = np.zeros((1,7))

n = 5 
b = 1
ysq = np.linspace(b,max(y)-b,n)
zsq = np.linspace(min(z)+b,max(z)-b,n)
xsq = np.max(x)*np.ones(1)
Xsq, Ysq, Zsq = np.meshgrid(xsq,ysq,zsq)
xpr = Xsq.ravel()
ypr = Ysq.ravel()
zpr = Zsq.ravel()
pr = np.c_[xpr,ypr,zpr]

for i in range(len(pr)):
    
    # Starting angle
    theta = 0
    
    # Starting point
    x0 = pr[i,0]
    y0 = pr[i,1]
    z0 = pr[i,2]
    
    r_0 = [x0,y0,z0]
    
#    idx, idy, idz, t00, n00 = N3Dc(r_0)
    
    # Starting velocity
    v_0 = start_point(x,y,z,r_0,theta)
    
    # Integration
    sol3d = odeint(diff_y3d,r_0 + v_0,t_range)
    
    # Export
    rcur, dx, dy, dz = exp_res(sol3d,x,y,z,nx,ny,nz,r_0)
    rall = np.vstack((rall,rcur))
    xd.append(x0)
    yd.append(y0)
    zd.append(z0)
    delx.append(dx/1000)
    dely.append(dy/1000)
    delz.append(dz/1000)
    
    print('%d / %d' % ((i+1),len(pr)) + ', %d' % (100*(i+1)/len(pr)) + ' %')

# Export 3D Ray trajectory
rall = np.delete(rall, (0), axis=0)
rall[:,2] = rall[:,2] - offset
#head2 = 'TITLE = "3D XYZ"\nVARIABLES = "X","Y","Z"\nZONE T = "Frame 1", I=%d, J=%d' % (len(xout),3)
#np.savetxt('3D_XYZ.dat',rto,fmt=['%6.4e','%6.10e','%6.10e'],delimiter='\t',header=head2,comments='')
head2 = 'X,Y,Z,dXp,dYp,dZp,dTOTp'
np.savetxt(direc + '/' + 'KF_3D_XYZ.csv',rall,fmt=['%8.4f','%8.4f','%8.4f','%10.6e','%10.6e','%10.6e','%10.6e'],delimiter=',',header=head2,comments='')

# Export 3D Ray deflection
deltot = []
for i in range(len(delx)):
    deltot.append(np.sqrt(dely[i]**2+delz[i]**2))
zd = [zd[i]-offset for i in range(len(zd))]
defl = np.c_[range(1,len(zsq)*len(ysq)+1),xd,yd,zd,delx,dely,delz,deltot]
head3 = 'N,X0,Y0,Z0,dX,dY,dZ,dTOT'
np.savetxt(direc + '/KF_3D_Deflection.csv',defl,fmt='%10.6f',delimiter=',',header=head3,comments='')

#%% Plot

'''

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pl1 = 0
pl2 = 0
pl3 = 0

if pl1 == 1:
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    cmm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=np.min(t3d), vmax=np.max(t3d))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmm)
    nnn = 2
    ax.scatter(fo[::nnn,0],fo[::nnn,1],fo[::nnn,2], c=scalarMap.to_rgba(fo[::nnn,3]), alpha=0.5)
    scalarMap.set_array(t3d)
    fig.colorbar(scalarMap)
    ax.set_xlim([np.min(x3),np.max(x3)])
    ax.set_ylim([np.min(y3),np.max(y3)])
#    ax.set_zlim([np.min(z3),np.max(z3)])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.show()

#if pl2 == 0:
#    
#    foo = fo[fo[:,0].argsort()]
#
#    npp = 100
#    xp = foo[::npp,0]
#    yp = foo[::npp,1]
#    zp = foo[::npp,2]
#    npl = foo[::npp,3]
#    cs = npl
#    
#    fig = plt.figure(1)
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot(rx, ry, rz, 'y',linewidth=2)
#    ax.quiver(r_0[0],r_0[1],r_0[2],v_0[0],v_0[1],v_0[2])
#    ax.scatter(r_0[0],r_0[1],r_0[2],color='r')
#    cmm = plt.get_cmap('jet')
#    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
#    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmm)
#    ax.scatter(xp,yp,zp, c=scalarMap.to_rgba(cs), alpha=0.3)
#    scalarMap.set_array(cs)
#    fig.colorbar(scalarMap)
#    ax.set_xlim([np.min(x),np.max(x)])
#    ax.set_ylim([np.min(y),np.max(y)])
#    ax.set_zlim([np.min(z),np.max(z)])
#    ax.set_xlabel('X [mm]')
#    ax.set_ylabel('Y [mm]')
#    ax.set_zlabel('Z [mm]')
#    plt.show()
#
#if pl3 == 0:
#    
#    plt.plot(range(n),dely,'r',label='dY')
#    plt.plot(range(n),delz,'k',label='dZ')
#    plt.xlabel('starting Y [mm]')
#    plt.ylabel('Deflection [mm]')
#    plt.legend()
#    plt.grid()
#
'''