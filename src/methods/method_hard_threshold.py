from methods.benchmark_utils import MethodTemplate
import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import *

def hard_thresholding(signal, coeff=3, dict_output=False):
    
    Nfft = len(signal)
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


    def method(self, signal, *args, **kwargs):
        signal_output = hard_thresholding(signal, *args, **kwargs)          
        return signal_output