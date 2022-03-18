from methods.MethodTemplate import MethodTemplate
import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import *

def hard_thresholding(signal, coeff=3, dict_output=False):
    if len(signal.shape) == 1:
        signal = np.resize(signal,(1,len(signal)))

    Nfft = signal.shape[1]
    g, a = get_round_window(Nfft)
    S, stft, stft_padded, Npad = get_spectrogram(signal,g)

    gamma = np.median(np.abs(np.real(stft)))/0.6745
    thr = coeff*np.sqrt(2)*gamma
    mask = np.abs(stft)
    mask[mask<thr] = 0
    mask[mask>=thr] = 1

    # mask[:] = 1
    xr, t = reconstruct_signal_2(mask, stft_padded, Npad)

    if dict_output:
        return {'xr': xr, 'mask': mask} 
    else:
        return xr


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'hard_thresholding'
        self.task = 'denoising'


    def method(self, signals, params):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            if params is None:
                signals_output[i] = hard_thresholding(signal)
            else:
                signals_output[i] = hard_thresholding(signal, **params)          

        return signals_output
