# -*- coding: utf-8 -*-
"""
Created on 19-06-2018

******************
*** INCOMPLETE ***
******************

This file generates a .wav file of multiple signals.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy.io.wavfile as sp

# import matplotlib for plotting
import matplotlib.pyplot as plt

#%% Create signals

# sampling frequency
sf = 250000

# time between samples
n = 1/sf

# total time
tt = 10

x = np.arange(0, tt, n)

# signal frequency
sigf = 5

# signal 1
f1 = 2*np.pi*sigf
ph1 = 0
ns1 = np.random.rand(len(x))/10
y1 = np.sin(f1*x - ph1*sf*n) + ns1

# signal 2
f2 = 2*np.pi*sigf
ph2 = 1/2*np.pi
ns2 = np.random.rand(len(x))/10
y2 = np.sin(f2*x - ph2*sf*n) + ns2

# signal 3
f3 = 2*np.pi*sigf
ph3 = np.pi
ns3 = np.random.rand(len(x))/10
y3 = np.sin(f3*x - ph3*sf*n) + ns3

# signal 4
f4 = 2*np.pi*sigf
ph4 = 3/2*np.pi
ns4 = np.random.rand(len(x))/10
y4 = np.sin(f4*x - ph4*sf*n) + ns4

#%% Plot signals

fig = plt.figure(figsize=(16,6))

# plotting increment for faster computation
npl = 100
plt.plot(x[::npl], y1[::npl], x[::npl], y2[::npl], x[::npl], y3[::npl], \
         x[::npl], y4[::npl])
plt.xlim([0, tt/10])
plt.grid()
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.legend(['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4'], loc=3)

#%% Export data

# output data
out = np.c_[y1, y2, y3, y4]
out = np.asarray(out, dtype=np.float32)

# filename
fn = 'cali_t_%d_f1_%d_ph1_%.3f_f2_%d_ph2_%.3f_f3_%d_ph3_%.3f_f4_%d_ph4_%.3f' \
% (tt, f1, ph1, f2, ph2, f3, ph3, f4, ph4)

# save WAV file
#sp.write(fn + '.wav', sf, out)

# save image
#fig.savefig(fn + '.png')

# save CSV file
#np.savetxt(fn + '.csv', out, fmt='%.4e', delimiter=',', header='"SEP=,"', \
#comments='')
