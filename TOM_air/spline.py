# -*- coding: utf-8 -*-
"""

This file fits data from an Origin worksheet using a spline fit and stores the
output in a new worksheet.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
from scipy.signal import savgol_filter as svg
from scipy.interpolate import UnivariateSpline as spl
import scipy.fftpack

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import sys for basic system operations
import sys

# import pandas to read in external data
import pandas as pd

# import tkinter for file selection dialog
import tkinter
tkinter.Tk().withdraw()
from tkinter.filedialog import askopenfilename as ask

#%% Import data

# Filename(s)
inp = ask()

# Import data
data = pd.read_csv(inp, header=1, names=['Time','Width','Temperature'])

# Extract x and y variables
x = np.int32(data['Time'])
y1 = np.float64(data['Width'])
y2 = np.float64(data['Temperature'])

#%% Spline fit function

# Spline fit function (x input, y input, min x, max x, number of points)
def spline_fit(x,y,xmin,xmax,n):
        
    # Slice out region
    xc = x[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]]
    yc = y[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]] 
   
    # Compute spline fit
    if (xmax+n) > len(x):
        print('\n')
        print('NUMBER OF POINTS EXCEEDS INPUT SIZE')
        print('\n')
        sys.exit()
    else:
        
        # Initialize output variables
        xcur = np.zeros(n)
        ycur = np.zeros(n)
        yspl = []
        
        # Compute fit
        for i in np.arange(0,len(xc)+n,n):
        
            # Slice out n points per step, compute third order cubic spline
            xcur = x[np.nonzero(x==xmin)[0][0]+i:np.nonzero(x==xmin)[0][0]+i+n]
            ycur = y[np.nonzero(x==xmin)[0][0]+i:np.nonzero(x==xmin)[0][0]+i+n]
            ys = spl(xcur,ycur, k=3)
            if i == len(xc)-1:
                yspl.extend(ys(xcur)[0])
                break
            else:
                yspl.extend(ys(xcur))
        
        # Store data
        yspl = np.float64(yspl[:len(xc)])
        
        # Return output
        return xc, yc, yspl

#%% Savicky Golay Function
        
def savgol(x,y,xmin,xmax,n,p):
    
    # Slice out region
    yc = y[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]]
    
    # Compute filtered output
    ysg = svg(yc,n,p)
    
    return ysg

#%% Moving average

def moving_average(x,y,xmin,xmax,N):
    
    # Slice out region
    yc = y[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]+N-1]
    cumsum = np.cumsum(np.insert(yc, 0, 0)) 
        
    return (cumsum[N:] - cumsum[:-N]) / N

#    window = np.ones(int(N))/float(N)
#    interval = yc
#    out = np.convolve(interval, window, 'same')
#    
#    return out[2:-2]

#%% Noise function

# Noise function (y original, y fitted, conversion factor)
def noise(y1,y2,c):
    
    # Initialize output variable
    out = []
    
    # Compute noise
    for i in range(len(y1)):
        out.append((y1[i]-y2[i])*c)
    
    # Store data and standard deviation
    out = np.float64(out)
    std = np.std(out)
    
    # Return output
    return out, std
    
#%% FFT function

# FFT function (input signal)
def fft_fit(y):
    
    # Number of sample points
    N = len(y)
    # Spacing between points
    T = 1.0/N
    
    # Compute FFT
    xfft = np.linspace(0.0, 1/(2*T), N/2)
    yfft = scipy.fftpack.fft(y)
    yfft_p = (2.0/N)*np.abs(yfft[:N//2])
    
    # Return output
    return xfft, yfft, yfft_p

#%% Plot function

# Plot function (x input, y1 input, y2 input, x sliced, y sliced, y fitted, standard deviation, noise, x FFT, y FFT for plot)
def plot1(x,y1,y2):
    
    ## 1
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,y1,'r')
    ax2.plot(x,y2,'b')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Width', color='r')
    ax2.set_ylabel('Temperature', color='b')
    ax1.grid(which='major', axis='both')
    ax1.set_xlim([min(x),max(x)])
    plt.show()

def plot2(x,y,y_fit,std,noi):

    ## 2
    plt.figure()
    plt.subplot(211)
    plt.plot(x,y,'r',label='Input')
    plt.plot(x,y_fit,'k',label=('Output, STD: %4.3f' % std))
    plt.xlim([min(x),max(x)])
    plt.grid()
    plt.legend(loc=1)
    
    plt.subplot(212)
    plt.plot(x,noi,'r',label='Noise')
    plt.grid()
    plt.xlim([min(x),max(x)])
    plt.legend(loc=1)

def plot3(x,y,noi):

    ## 3
    plt.figure()
    plt.subplot(211)
    plt.plot(x,noi,'r')
    plt.grid()
    plt.subplot(212)
    plt.plot(x,y,'r')
    plt.grid()
    plt.xlim([min(x),max(x)])

#%% Run code

'''
First 1400 K hold:  280 - 398
1300 K hold:        418 - 538
Second 1400 K hold: 559 - 678
1000 K hold:        759 - 878
'''

x1 = 759
x2 = 878
n = 5
p = 2
c = 1014.0

# Spline fit function (x input, y input, min x, max x, number of points)
x_sl, y_sl, y_sp = spline_fit(x,y1,x1,x2,n)

# Savitzky-Golay output
y_sg = savgol(x,y1,x1,x2,n,p)

# Moving average
y_ma = moving_average(x,y1,x1,x2,n)

# Noise function (y original, y fitted, conversion factor)
noi_sp, std_sp = noise(y_sl,y_sp,c)
noi_sg, std_sg = noise(y_sl,y_sg,c)
noi_ma, std_ma = noise(y_sl,y_ma,c)

# FFT function (input signal)
#xfft_sp, _, yfft_sp = fft_fit(noi_sp)
 
# Plot function (x input, y input, x sliced, y sliced, y fitted, standard deviation, noise, x FFT, y FFT for plot)
plot1(x,y1,y2)
plot2(x_sl,y_sl,y_sp,std_sp,noi_sp)
plot2(x_sl,y_sl,y_sg,std_sg,noi_sg)
plot2(x_sl,y_sl,y_ma,std_ma,noi_ma)
#plot3(xfft_sp,yfft_sp)
