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
import scipy as sp
import scipy.signal as sps
import scipy.io.wavfile as wav

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

# create signal function
def signal(x,freq,ph,ns,samf,samp):
    
    # noise
    if ns == 1:
        ns = np.random.rand(len(x))/10
    else:
        ns = 0
        
    sig = np.sin(2*np.pi*freq*x - ph*samf*samp) + ns
    
    return sig, ns
    
# signal 1
s1, ns1 = signal(x,sigf,0,1,sf,n)

# signal 2
s2, ns2 = signal(x,sigf,1/2*np.pi,1,sf,n)

# signal 3
s3, ns3 = signal(x,sigf,np.pi,1,sf,n)

# signal 4
s4, ns4 = signal(x,sigf,3/2*np.pi,1,sf,n)

#%% Plot signals

def plotsig(x,y1,y2,y3,y4,tt):
    
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

plotsig(x, s1, s2, s3, s4, tt)

#%% Filter signals

def butter_lowpass(cutoff, fs, order):
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sps.butter(order, normal_cutoff, btype='low', analog=False)
    
    return b, a

def butter_lowpass_filter(data, b, a):

    y = sps.lfilter(b, a, data)
    
    return y

def sigfilt(x,y,o,samf,fc):
    
    # cutoff frequency
    b, a = butter_lowpass(fc,samf,o)
    
    # Plot the frequency response.
    w, h = sps.freqz(b, a)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(0.5*samf*w/np.pi, np.abs(h), 'b')
    plt.plot(fc, 0.5*np.sqrt(2), 'ko')
    plt.axvline(fc, color='k')
#    plt.xlim(0, 0.5*samf)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    
    fsig = butter_lowpass_filter(y, b, a)
    
    plt.figure()
    plt.plot(x, y, 'k')
    plt.plot(x, fsig, 'r')
    plt.legend(['Original', 'Filtered'], loc=3)    
    plt.grid()
    
    return fsig

fs1 = sigfilt(x,s1,100,sf,2000)

#%% Export data

# output data
out = np.c_[s1, s2, s3, s4]
out = np.asarray(out, dtype=np.float32)

# filename
#fn = 'cali_t_%d_f1_%d_ph1_%.3f_f2_%d_ph2_%.3f_f3_%d_ph3_%.3f_f4_%d_ph4_%.3f' \
#% (tt, f1, ph1, f2, ph2, f3, ph3, f4, ph4)

# save WAV file
#wav.write(fn + '.wav', sf, out)

# save image
#fig.savefig(fn + '.png')

# save CSV file
#np.savetxt(fn + '.csv', out, fmt='%.4e', delimiter=',', header='"SEP=,"', \
#comments='')
