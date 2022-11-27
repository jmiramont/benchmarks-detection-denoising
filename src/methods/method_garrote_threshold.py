from benchmark_demo.benchmark_utils import MethodTemplate
import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import *

def garrote_thresholding(signal, coeff=1, dict_output=False):
    Nfft = len(signal)
    g, a = get_round_window(Nfft)
    S, stft, stft_padded, Npad = get_spectrogram(signal,g)

    gamma = np.median(np.abs(np.real(stft)))/0.6745
    thr = coeff*np.sqrt(2)*gamma

    aux = np.abs(stft)
    aux[aux<=thr] = 0
    aux[aux>thr] = aux[aux>thr]-(thr**2/aux[aux>thr])
    aux = aux / np.abs(stft)

    xr, t = reconstruct_signal_2(aux, stft_padded, Npad)

    if dict_output:
        return {'xr': xr, 'mask': aux} 
    else:
        return xr


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'garrote_thresholding'
        self.task = 'denoising'


    def method(self, signal, *args, **kwargs):
        signal_output = garrote_thresholding(signal, *args, **kwargs)          
        return signal_output
