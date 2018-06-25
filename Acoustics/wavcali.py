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
f_samp = 250000

# sampling period
t_samp = 1/f_samp

# total time
t_tot = 5

# digital time
x = np.arange(0, t_tot, t_samp)

# signal amplitude and frequency
a_sig = 1
f_sig = 500

# noise amplitude and frequency
a_noi = 1/10
f_noi = 25000


# create signal function
def signal(x, amp_signal, freq_signal, phase_signal, noise, amp_noise,
           freq_noise, freq_samp, per_samp):

    # noise
    if noise == 2:
        ns = a_noi*np.random.rand(len(x))
    elif noise == 1:
        ns = a_noi*np.cos(2*np.pi*f_noi*x)
    else:
        ns = 0

    # output signal
    sig = amp_signal*np.sin(2*np.pi*freq_signal*x - phase_signal*freq_samp*
                            per_samp) + ns

    return sig, ns

# noise switch
nss = 2

# signal 1
s1, ns1 = signal(x, a_sig, f_sig, 0, nss, a_noi, f_noi, f_samp, t_samp)

# signal 2
s2, ns2 = signal(x, a_sig, f_sig, 1/2*np.pi, nss, a_noi, f_noi, f_samp, t_samp)

# signal 3
s3, ns3 = signal(x, a_sig, f_sig, np.pi, nss, a_noi, f_noi, f_samp, t_samp)

# signal 4
s4, ns4 = signal(x, a_sig, f_sig, 3/2*np.pi, nss, a_noi, f_noi, f_samp, t_samp)

#%% Plot signals


# define function to plot all signals
def plotsig(x, y1, y2, y3, y4, time):

    plt.figure(figsize=(16, 6))

    # plotting increment for faster computation
    npl = 1
    plt.plot(x[::npl], y1[::npl], 'k')
    plt.plot(x[::npl], y2[::npl], 'r')
    plt.plot(x[::npl], y3[::npl], 'b')
    plt.plot(x[::npl], y4[::npl], 'g')
    plt.xlim([0, time])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4'], loc=3,
               fontsize='large', edgecolor='k', framealpha=1)


# define function to plot transformed signals
def plotfft(x1, mag1, ph1, x2, mag2, ph2, x3, mag3, ph3, x4, mag4, ph4, time):

    fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8] = plt.subplots(8, 1,
         sharex=True, figsize=(16, 6),)

    ax1.plot(x1, mag1, 'k')
    ax1.set_yticks([])
    ax1.legend('1')
    ax1.grid(True)

    ax2.plot(x1, ph1, 'k')
    ax2.set_yticks([])
    ax2.grid(True)

    ax3.plot(x2, mag2, 'r')
    ax3.set_yticks([])
    ax3.legend('2')
    ax3.grid(True)

    ax4.plot(x2, ph2, 'r')
    ax4.set_yticks([])
    ax4.grid(True)

    ax5.plot(x3, mag3, 'b')
    ax5.set_yticks([])
    ax5.legend('3')
    ax5.grid(True)

    ax6.plot(x3, ph3, 'b')
    ax6.set_yticks([])
    ax6.grid(True)

    ax7.plot(x4, mag4, 'g')
    ax7.set_yticks([])
    ax7.legend('4')
    ax7.grid(True)

    ax8.plot(x4, ph4, 'g')
    ax8.set_yticks([])
    ax8.grid(True)

    plt.subplots_adjust(hspace=0.0)

    plt.xlim([0, np.max([x1, x2, x3, x4])])


#%% transform signals from time to frequency domain

# https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
# function to compute fourier transform
def fft(t_samp, signal):

    # compute fft
    fft = np.fft.fft(signal)

    # signal length, spacing, and frequency
    N = signal.size
    T = t_samp
    f = np.linspace(0, 1/T, N)
    x = f[:N//2]

    # magnitude
    mag = np.abs(fft)*(1/N)

    # clean signal for phase calculation
    fft_thresh = fft
    thresh = np.max(np.abs(fft))/10000
    fft_thresh[np.abs(fft_thresh) < thresh] = 0

    # phase
    phase = np.arctan2(np.imag(fft_thresh), np.real(fft_thresh))*(180/np.pi)

    return x, mag[:N//2], phase[:N//2]


s1_fftx, s1_fftmag, s1_fftph = fft(t_samp, s1)
s2_fftx, s2_fftmag, s2_fftph = fft(t_samp, s2)
s3_fftx, s3_fftmag, s3_fftph = fft(t_samp, s3)
s4_fftx, s4_fftmag, s4_fftph = fft(t_samp, s4)

#%% Compare original and transformed signals

plt.close('all')

# plot original signals
plotsig(x, s1, s2, s3, s4, t_tot/f_sig)

# plot transformed signals
plotfft(s1_fftx, s1_fftmag, s1_fftph, s2_fftx, s2_fftmag, s2_fftph, s3_fftx,
        s3_fftmag, s3_fftph, s4_fftx, s4_fftmag, s4_fftph, s1_fftx)

#%% Filter signals

# =============================================================================
#
# def butter_lowpass(cutoff, fs, order):
#
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = sps.butter(order, normal_cutoff, btype='low', analog=False)
#
#     return b, a
#
#
# def butter_lowpass_filter(data, b, a):
#
#     y = sps.lfilter(b, a, data)
#
#     return y
#
#
# def sigfilt(x, y, o, samf, fc):
#
#     # cutoff frequency
#     b, a = butter_lowpass(fc, samf, o)
#
#     # Plot the frequency response.
#     w, h = sps.freqz(b, a)
#
#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot(0.5*samf*w/np.pi, np.abs(h), 'b')
#     plt.plot(fc, 0.5*np.sqrt(2), 'ko')
#     plt.axvline(fc, color='k')
# #    plt.xlim(0, 0.5*samf)
#     plt.title("Lowpass Filter Frequency Response")
#     plt.xlabel('Frequency [Hz]')
#     plt.grid()
#
#     fsig = butter_lowpass_filter(y, b, a)
#
#     plt.figure()
#     plt.plot(x, y, 'k')
#     plt.plot(x, fsig, 'r')
#     plt.legend(['Original', 'Filtered'], loc=3)
#     plt.grid()
#
#     return fsig
#
#
# fs1 = sigfilt(x, s1, 100, f_samp, 2000)
#
# =============================================================================
#%% Export data
# =============================================================================
#
# # output data
# out = np.c_[s1, s2, s3, s4]
# out = np.asarray(out, dtype=np.float32)
#
# # filename
# #fn = 'cali_t_%d_f1_%d_ph1_%.3f_f2_%d_ph2_%.3f_f3_%d_ph3_%.3f_f4_%d_ph4_%.3f' \
# #% (t_tot, f1, ph1, f2, ph2, f3, ph3, f4, ph4)
#
# # save WAV file
# #wav.write(fn + '.wav', f_samp, out)
#
# # save image
# #fig.savefig(fn + '.png')
#
# # save CSV file
# #np.savetxt(fn + '.csv', out, fmt='%.4e', delimiter=',', header='"SEP=,"', \
# #comments='')
# =============================================================================
