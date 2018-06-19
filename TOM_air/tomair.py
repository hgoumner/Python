# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:15:20 2017

This file reads in data and adds a 'window' representing the sapphire window in
TOM_air.

@author: Hristo Goumnerov
"""
#%% Import modules
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import tkinter
from tkinter.filedialog import askopenfilenames as asks
tkinter.Tk().withdraw()

#%% Inputs

# Filename(s)
fns = asks()

base = []

#%% Load input
    
for i in range(len(fns)):
    
    base.append(os.path.basename(fns[i])[:-4])
    
    if i == 0:
        data = np.zeros((len(fns),100,5))
    
        with open(fns[i], 'r') as file:
            N = 20
            for j in range(N):
                print(str(j+1) + '. ' + file.readline())
            fl = len(file.readlines())+N
        h = int(input('Last line of header: '))
    
    cur = pd.read_csv(fns[i], sep=',', header=h-1)
    cur = cur.sort_values(cur.columns[0])
    data[i,:cur.shape[0],:] = cur
#    data[i] = data[i][~(data[i]==0).all(1)]

#%% Plot function

def pll(x,T,u,w,name):
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(x,T-ktoc,'bo-',label=['Temp ' + name])
    ax1.plot(xw1,yw1,'k',xw2,yw2,'k')
    ax1.set_ylim([0,2000])
    ax1.set_xlim([0,450])
    ax1.xaxis.set_ticks(np.arange(0, 450, 50))
    ax1.legend(loc=1)
    ax1.grid()
    
    ax2.plot(x,u,'b',label=['u ' + name])
    ax2.plot(x,w,'r',label=['w ' + name])
    ax2.plot(xw1,yw1,'k',xw2,yw2,'k')
    ax2.set_ylim([-0.5,0.5])
    ax2.legend(loc=1)
    ax2.grid()

# Window
xw1 = 131*np.ones(10)
xw2 = 158*np.ones(10)

yw1 = np.linspace(-3000,3000,10)
yw2 = np.linspace(-3000,3000,10)

#%% Run

# Offset
ktoc = 273.15

for k in range(len(fns)):
#    pll(data[k,:,0],data[k,:,2],data[k,:,3],data[k,:,4],str(base[k]))
    outd = data[k,:,:][~(data[k,:,:]==0).all(1)]
#   
    np.savetxt(fns[k][:-4] + '_data.csv', outd, delimiter=',', fmt='%10.4f')
    
f, (ax1, ax2) = plt.subplots(2, sharex=True)

for l in range(4):
    
    ax1.plot(data[l,:,0],data[l,:,2]-ktoc,'g',label=['Temp ' + str(base[l])])
    ax2.plot(data[l,:,0],data[l,:,3],'b',label=['u ' + str(base[l])])
    ax2.plot(data[l,:,0],data[l,:,4],'r',label=['w ' + str(base[l])])

ax1.plot(xw1,yw1,'k',xw2,yw2,'k')
ax1.set_ylim([0,2000])
ax1.set_xlim([0,450])
ax1.xaxis.set_ticks(np.arange(0, 450, 50))
ax1.legend(loc=1)
ax1.grid()

ax2.plot(xw1,yw1,'k',xw2,yw2,'k')
ax2.set_ylim([-0.5,0.5])
ax2.legend(loc=1)
ax2.grid()

