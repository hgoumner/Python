# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:06:34 2018

This file fits the temperature data measured by the pyrometer according to the
scale of the AOM.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import matplotlib for plotting
import matplotlib.pyplot as plt

#%% Import data

# Offset
offs = np.loadtxt('voff.txt',skiprows = 1,delimiter=',')
offs[:,0] = offs[:,0]+273.15
offs = offs[6:,:]

# AOM
aom = np.loadtxt('aom.csv',skiprows = 1,delimiter=';')
xaom = np.arange(0,900,0.5)
yaom = np.interp(xaom,aom[:,0]*90,aom[:,1])

plt.figure()
plt.plot(xaom,yaom)

# Data
data = np.loadtxt('3mm400500.csv',skiprows = 0,delimiter=';')
data = data[13:,:]
data[:,0] = data[:,0] - data[0,0]
#data[:,1] = data[:,1]*200/np.max(data[:,1])
data[1:len(xaom)+1,1] = yaom*2 
t = data[:,0]
pin = data[:,1]
idx = np.nonzero(pin>0)[0][-2]
pin[idx+1] = pin[idx]
vin = data[:,2]
#vin[vin<0.5] = 0.1633574719824192

plt.figure()
plt.plot(t,pin)

n = len(data)
vout = np.zeros(n)
check = np.zeros((n,3))

for i in range(n):
    
    pcur = pin[i]
    check[i,0] = vin[i]
    check[i,1] = pcur
    
#    off_int = np.interp(pcur,offs[:,1],offs[:,2])
    off_int = 15.56/498*pcur
    check[i,2] = off_int
    
    vout[i] = data[i,2] - off_int
    
    if vout[i] < 0.1:
        vout[i] = 0.1633574719824192
    
# Fitting from T to V
a = 8.15922e-08
b = 2.55422

xttov = np.arange(300,1400,10)

yttov = a*xttov**b

plt.figure()
plt.plot(xttov,yttov)

# Fitting from V to T

tin = (vin/a)**(1/b)-273.15
tout = (vout/a)**(1/b)
tout[tout<293.15] = 293.15
tout = tout - 273.15

plt.figure()
plt.plot(t,vin,'r',t,vout,'k')

plt.figure()
plt.plot(t,tin,'r',t,tout,'k')

out = np.c_[t,vin,tin,vout,tout]

np.savetxt('3mm400500_mod.csv', out, fmt='%10.6f', delimiter=',')
