# -*- coding: uTf-8 -*-
"""
CreaTed on %(daTe)s

@auThor: %(username)s
"""
#%% Import modules

# pylint: disable=I0011,C0103,E1101
import sys

import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import matplotlib.pyplot as plt

import pandas as pd
#%% Import Data

# Load Pyrometer Data
pyr = pd.read_csv("pyro.csv", header=1)
pyro = np.delete(np.float64(pyr), 1, axis=1)

# Assign Temperature and Voltage variables
offs = 273.15
Temp = np.float64(pyro[:,0]+offs)
V = np.float64(pyro[:,1])

#plt.plot(Temp, V)
#%% Fit mapping function

p = 1
e = 1

if p==1:
    ##################
    ### Polynomial ###
    ##################
    
    # Temp to V mapping
    deg = 4
    
    polyTV = np.polyfit(Temp, V, deg)
    mappolyTV = np.poly1d(polyTV)
    
    def funcp(Tin, pr):
        return mappolyTV(Tin) - pr
    
    # Y To plot Temp To V mapping
    ypolyTV = funcp(Temp, 0)
    
    # Accuracy
    _, _, rp, _, _ = sps.linregress(V, ypolyTV)

if e==1:
    ###################
    ### Exponential ###
    ###################
    
    def funce(Tin,a,b):
        return a*(Tin)**b
    
    popt, pcov = spo.curve_fit(funce,Temp,V)
    
    yexpTV = funce(Temp, *popt)
    
    # Accuracy
    residuals = V-funce(Temp, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((V-np.mean(V))**2)
    re = 1-(ss_res/ss_tot)

###################
###### Plot #######
###################

if p==1:
    # Polynomial
    #print(mappolyTV)
    plt.plot(Temp, V, 'ko-', Temp, ypolyTV, 'r')
    plt.title('Polynomial')
    plt.legend(['Original', 'Fit, R^2 = %4.4f' % rp])
    plt.xlabel('Temp [K]')
    plt.ylabel('V [V]')

if e==1:
    # Exponential
    plt.figure()
    plt.plot(Temp, V, 'ko-', Temp, yexpTV, 'r')
    plt.title('Exponential')
    plt.legend(['Original', 'Fit, R^2 = %4.4f' % re])
    plt.xlabel('Temp [K]')
    plt.ylabel('V [V]')
#%% Import Temperature Data

# Load Temperature Data
Temperature = pd.read_csv('400W.csv', header=8)

# Assign Time variable
t = list(Temperature.columns.values)
t = t[2:]
t = np.array([int(i) for i in t])

# Assign Radius and Area
R = np.float64(Temperature['% R'])
rin = float(input('Radius: '))
if rin < min(R) or rin > max(R):
#    print('\n'*50)
    print('WRONG RADIUS. RUN AGAIN.\n\n')
    sys.exit()

Rc = ((R-rin)<0).argmin()+1
r = R[:Rc]
A = np.pi*max(r)**2
At = np.pi*max(R)**2

# Allocate original Temperature variables
Tcur = np.zeros([Temperature.shape[0], 1])
T_ori = np.zeros([len(t), 1])

# Allocate Mapping variables
# Polynomial
Up = np.zeros([len(r), 1])
Ubarp = np.zeros([len(t), 1])
T_mapp = np.zeros([len(t), 1])

## Exponential
Ue = np.zeros([len(r), 1])
Ubare = np.zeros([len(t), 1])
T_mape = np.zeros([len(t), 1])

# Create subplots for U and Tcur plots
f, (ax1, ax2) = plt.subplots(2, sharex=True)
Tmax = Temperature.max().max()+offs
Umax = mappolyTV(Tmax)

# Map Temperature
for j in range(len(t)):
    
    # Load Temperature for current time
    Tcur = np.float64(Temperature[Temperature.columns[j+2]]+offs)
    
    ###### Original Temperature ######
    ##################################
    
    T_ori[j] = np.trapz(2*np.pi*r*Tcur[:len(r)], r)/A
#    T_ori[j] = np.trapz(2*np.pi*R*Tcur, R)/At
    
    for k in range(len(r)):
    
        ####### Mapped Temperature #######
        ##################################
        if p==1:
            ### Polynomial ###
            # Convert T to V
            Up = funcp(Tcur[:len(r)], 0)
            
            # Integrate and average V
            Ubarp[j] = np.trapz(2*np.pi*r*Up, r)/A
            
            # Convert average V to average T
            invp = np.roots(mappolyTV - Ubarp[j])
        
            if deg>2:
                T_mapp[j] = float(invp[np.logical_and(invp.imag==0,invp.real<=max(Temp))].real)
            elif deg==2:
                T_mapp[j] = invp[1]
            else:
                T_mapp[j] = float(invp)
        
        if e==1:
            ### Exponential ###
            # Convert T to V
            Ue = funce(Tcur[:len(r)],*popt)
            
            # Integrate and average V 
            Ubare[j] = np.trapz(2*np.pi*r*Ue, r)/A
        
            # Convert average V to averate T
            T_mape[j] = float((Ubare[j]/popt[0])**(1/popt[1]))
    
    # Plot U and Tcur for specified times
    if t[j] % 100 == 0:
        ax1.plot(r, Tcur[:len(r)])
        ax1.plot(R, Tcur, 'k--', linewidth=0.4)
        if Up[4]==0:
            ax2.plot(r, Ue, label=str(t[j])+' s')
            ax2.set_ylabel('Ue')
        else:
            ax2.plot(r, Up, label=str(t[j])+' s')
            ax2.set_ylabel('Up')
        ax1.set_ylabel('Temp')      
        legend = ax2.legend(loc='lower right')
        legend.get_frame().set_alpha(1)
#        legend.get_frame().set_facecolor('peachpuff')
#        ax1.set_ylim([0,1]), ax2.set_ylim([0,1])

# Allocate output data
if p==1 and e==1:
    output = np.c_[t, T_ori, T_mapp, T_mape]
elif p==1 and e==0:
    output = np.c_[t, T_ori, T_mapp]
elif p==0 and e==1:
    output = np.c_[t, T_ori, T_mape]

#%% Export Data

exp = 1

if exp==1:
    
    # Open file
    f = open('400W_int.csv','w')
    
    # Write header
    f.write('Time (s), Temperature (K)\n')
    
    # Write data
    for l in range(len(output)):
        f.write(str(output[l,0]) + ',' + str(output[l,-1]) + '\n')
    
    # Close file
    f.close()
#%% Plot Data

if p==1 and e==1:
    plt.figure()
    plt.plot(output[:,0], output[:,1], 'k--', output[:,0], output[:,2], 'r', output[:,0], output[:,3], 'b')
    plt.legend(['T_int', 'Up_int', 'Ue_int'], loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
elif p==1 and e==0:
    plt.figure()
    plt.plot(output[:,0], output[:,1], 'k--', output[:,0], output[:,2], 'r')
    plt.legend(['T_int', 'Up_int'], loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
elif p==0 and e==1:
    plt.figure()
    plt.plot(output[:,0], output[:,1], 'k--', output[:,0], output[:,2], 'r')
    plt.legend(['T_int', 'Ue_int'], loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')