from benchmark_demo.benchmark_utils import MethodTemplate
import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import *

def garrote_thresholding(signal, coeff=1.5, window=None, Nfft=None, dict_output=False):
    if Nfft is None:
        Nfft = 2*len(signal)
    
    if window is None:    
        window, a = get_round_window(Nfft)
    
    _, stft_whole = get_spectrogram(signal,window=window,Nfft=Nfft)

    stft = stft_whole[0:Nfft//2+1,:]
    gamma = np.median(np.abs(np.real(stft)))/0.6745

    thr = coeff*np.sqrt(2)*gamma

    aux = np.abs(stft)
    aux[aux<=thr] = 0
    aux[aux>thr] = aux[aux>thr]-(thr**2/aux[aux>thr])
    aux = aux / np.abs(stft)

    xr = np.real(reconstruct_signal_3(aux, stft_whole, window=window))

    if dict_output:
        return {'xr': xr, 'mask': aux} 
    else:
        return xr


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'thresholding_garrote'
        self.task = 'denoising'


    def method(self, signal, *args, **kwargs):
        signal_output = garrote_thresholding(signal, *args, **kwargs)          
        return signal_output
