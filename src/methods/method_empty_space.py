from methods.MethodTemplate import MethodTemplate
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import scipy.stats as spst
import scipy.signal as sg
from benchmark_demo.utilstf import *


def find_center_empty_balls(Sww, pos_exp, a, radi_seg=1):
    # define a kd-tree with zeros
    kdpos = KDTree(pos_exp)

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0])/a)
    vecy = (np.arange(0, Sww.shape[1])/a)
    g = np.transpose(np.meshgrid(vecy, vecx))
    result = kdpos.query_ball_point(g, radi_seg).T

    empty_mask = np.zeros(result.shape, dtype=bool)
    for i in range(len(vecx)):
        for j in range(len(vecy)):
            empty_mask[i,j] = len(result[i, j]) < 1

    return empty_mask


def get_convex_hull(Sww, pos_exp, empty_mask, radi_expand=0.5):
    # extract region of interest
    Nfft = Sww.shape[1]
    fmin = int(np.sqrt(Nfft))//2
    fmax = empty_mask.shape[0] - fmin
    tmin = int(np.sqrt(Nfft))//2
    tmax = empty_mask.shape[1] - tmin
    sub_empty = empty_mask[fmin:fmax, tmin:tmax]
    vecx = (np.arange(0, sub_empty.shape[0]))
    vecy = (np.arange(0, sub_empty.shape[1]))
    g = np.transpose(np.meshgrid(vecx, vecy))
    u, v = np.where(sub_empty)
    kdpos = KDTree(np.array([u, v]).T)
    result = kdpos.query_ball_point(g, radi_expand*np.sqrt(Nfft))
    sub_empty = np.zeros(result.shape, dtype=bool)
    for i in range(sub_empty.shape[1]):
        for j in range(sub_empty.shape[0]):
            sub_empty[j, i] = len(result[j, i]) > 0

    u, v = np.where(sub_empty)
    points = np.array([u, v]).T
    hull_d = Delaunay(points) # for convenience
    mask = np.zeros(Sww.shape)
    mask[fmin:fmax, tmin:tmax] = sub_empty
    return hull_d, mask


def empty_space_denoising(signal, radi_seg=1.0, radi_expand=0.5, return_dic=False):
    if len(signal.shape) == 1:
        signal = np.resize(signal,(1,len(signal)))

    Nfft = signal.shape[1]
    g, a = get_round_window(Nfft)
    Sww, stft, stft_padded, Npad = get_spectrogram(signal,g)
    pos = find_zeros_of_spectrogram(Sww)
    pos_aux = pos.copy()
    pos_aux[:,0] = pos[:,1]/a
    pos_aux[:,1] = pos[:,0]/a
    empty_mask = find_center_empty_balls(Sww, pos_aux, a, radi_seg=radi_seg)
    hull_d , mask = get_convex_hull(Sww, pos_aux, empty_mask, radi_expand=radi_expand)
    xr, t = reconstruct_signal_2(mask, stft_padded, Npad)

    if return_dic:
        return {'s_r': xr,
                'mask': mask,
                'zeros':pos}
    else:
        return xr


class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'empty_space'
        self.task = 'denoising'
        

    def method(self, signals, params):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            if params is None:
                signals_output[i] = empty_space_denoising(signal)
            else:
                signals_output[i] = empty_space_denoising(signal, **params)    
        return signals_output
        