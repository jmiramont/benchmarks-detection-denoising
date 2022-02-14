import numpy as np
from numpy import pi as pi
import scipy.signal as sg

# from SignalBank import SignalBank
from matplotlib import pyplot as plt


# from utilstf import *

def hardThresholding(signal):
    Ni = len(signal)
    Npad = Ni//2

    signal_pad = np.zeros(Ni+2*Npad)
    signal_pad[Npad:Npad+Ni] = signal
    Nfft = Ni
    
    # analysis window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()

    # computing STFT
    _, _, stft = sg.stft(signal_pad, window=g, nperseg=Nfft, noverlap = Nfft-1)
    aux = stft[:,Npad:Npad+Ni]

    gamma = np.median(np.abs(np.real(aux)))/0.6745
    thr = 3*np.sqrt(2)*gamma
    mask = np.abs(stft)
    mask[mask<thr] = 0
    mask[mask>=thr] = 1

    t, xr = sg.istft(mask*stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    xr = xr[Npad:Npad+Ni]

    results = {'reconstruction': xr,
                'stft': stft,
                'mask': mask}
    return xr, results



def method_HT(signals, params):
    signals_output = np.zeros(signals.shape)
    for i,signal in enumerate(signals):
        signals_output[i], _ = hardThresholding(signal)
    return signals_output     


