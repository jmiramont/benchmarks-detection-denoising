import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sg
from math import atan2
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay


""" This file contains a number of utilities for time-frequency analysis.
Some functions has been modified from the supplementary code of:
Bardenet, R., Flamant, J., & Chainais, P. (2020). On the zeros of the spectrogram of white noise.
Applied and Computational Harmonic Analysis, 48(2), 682-705.
which can be found in: http://github.com/jflamant/2018-zeros-spectrogram-white-noise.

Those functions are:
- getSpectrogram(signal)
- findCenterEmptyBalls(Sww, pos_exp, radi_seg=1)
- getConvexHull(Sww, pos_exp, empty_mask, radi_expand=0.5)
- reconstructionSignal(hull_d, stft)

"""

def get_round_window(Nfft):
    # analysis window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    T = np.sqrt(Nfft)
    return g, T


def get_stft(signal, window):
    """ Compute the STFT of the signal. Signal is padded with zeros.
    The outputs corresponds to the STFT with the regular size and also the
    zero padded version.

    """

    N = np.max(signal.shape)
    Npad = N//2
    Nfft = len(window)
    signal_pad = np.zeros(N+2*Npad)
    signal_pad[Npad:Npad+N] = signal
    
    # computing STFT
    _, _, stft_padded = sg.stft(signal_pad, window=window, nperseg=Nfft, noverlap = Nfft-1)
    stft = stft_padded[:,Npad:Npad+N]
    return stft, stft_padded, Npad


def get_spectrogram(signal,window):
    """
    Get the round spectrogram of the signal 
    """
    stft, stft_padded, Npad = get_stft(signal, window)
    S = np.abs(stft)**2
    return S, stft, stft_padded, Npad


def find_zeros_of_spectrogram(S):
    # detection of zeros of the spectrogram
    th = 1e-14
    y, x = extr2minth(S, th) # Find zero's coordinates
    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = y
    pos[:, 1] = x
    # 2/15 Quedaron invertidos!!!!
    return pos


def find_center_empty_balls(Sww, pos_exp, radi_seg=1):
    Nfft = Sww.shape[1]
    # define a kd-tree with zeros
    kdpos = KDTree(pos_exp)

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0])/np.sqrt(Nfft))
    vecy = (np.arange(0, Sww.shape[1])/np.sqrt(Nfft))
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
    fmin = 1#int(np.sqrt(Nfft))
    fmax = empty_mask.shape[0] - fmin
    tmin = 1#int(np.sqrt(Nfft))
    tmax = empty_mask.shape[1] - tmin
    sub_empty = empty_mask[fmin:fmax, tmin:tmax]

    
    vecx = (np.arange(0, sub_empty.shape[0]))
    vecy = (np.arange(0, sub_empty.shape[1]))
    g = np.transpose(np.meshgrid(vecx, vecy))

    u, v = np.where(sub_empty)
    kdpos = KDTree(np.array([u, v]).T)
    result = kdpos.query_ball_point(g, radi_expand*np.sqrt(Nfft))

    # print(result.shape)

    sub_empty = np.zeros(result.shape, dtype=bool)
    for i in range(sub_empty.shape[1]):
        for j in range(sub_empty.shape[0]):
            sub_empty[j, i] = len(result[j, i]) > 0

    # plt.figure()
    # plt.imshow(sub_empty, origin = 'lower')

    u, v = np.where(sub_empty)
    points = np.array([u, v]).T
    # print(points.shape)
    
    hull_d = Delaunay(points) # for convenience
    mask = np.zeros(Sww.shape)
    mask[fmin:fmax, tmin:tmax] = sub_empty
    return hull_d, mask
   


def reconstruct_signal(hull_d, stft):
    """ Reconstruction using the convex hull
    """
    Nfft = stft.shape[1]
    tmin = int(np.sqrt(Nfft))
    tmax = stft.shape[1]-tmin
    fmin = int(np.sqrt(Nfft))
    fmax = stft.shape[0]-fmin

    # sub mask : check which points are in the convex hull
    vecx = (np.arange(0, stft.shape[0]-2*int(np.sqrt(Nfft))))
    vecy = (np.arange(0, stft.shape[1]-2*int(np.sqrt(Nfft))))
    g = np.transpose(np.meshgrid(vecx, vecy))
    sub_mask = hull_d.find_simplex(g)>=0
    mask = np.zeros(stft.shape)
    mask[fmin:fmax, tmin:tmax] = sub_mask
    print('mascara:{}'.format(mask.shape))
    # create a mask
    #mask = np.zeros(stft.shape, dtype=bool)
    #mask[fmin:fmax, base+tmin:base+tmax] = sub_mask

    # reconstruction
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    # t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask*stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    return mask, xr, t 

def reconstruct_signal_2(mask, stft, Npad):
    """ Reconstruction using a mask given as parameter
    """
    Ni = mask.shape[1]
    Nfft = Ni
    # reconstruction
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    mask_aux = np.zeros(stft.shape)
    mask_aux[:,Npad:Npad+Ni] = mask
    # t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask_aux*stft, window=g, nperseg=Nfft, noverlap = Nfft-1)
    xr = xr[Npad:Npad+Ni]
    return xr, t


def extr2minth(M,th):
    """ Finds the minima of the spectrogram matrix M
    """
    C,R = M.shape
    Mid_Mid = np.zeros((C,R), dtype=bool)
    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
            #Mid_Mid[c, r] = (np.min(T) == T[1, 1])
    x, y = np.where(Mid_Mid)
    return x, y


def snr_comparison(x,x_hat):
    qrf = 10*np.log10(np.sum(x**2)/np.sum((x-x_hat)**2))
    return qrf


def add_snr_block(x,snr,K = 1):
    """
    Adds noise to a signal x with SNR equal to snr. SNR is defined as SNR (dB) = 10 * log10(Ex/En)
    """
    N = len(x)
    x = x - np.mean(x)
    Px = np.sum(x ** 2)
    # print(x)

    n = np.random.rand(N,K)
    n = n - np.mean(n,axis = 0)
    # print(np.mean(n, axis = 0))
    # x = x+n

    Pn = np.sum(n ** 2, axis = 0)
    n = n / np.sqrt(Pn)
    # print(np.sum(n[:,0]**2))

    Pn = Px * 10 ** (- snr / 10)
    n = n * np.sqrt(Pn)
    snr_out1 = 20 * np.log10(np.sqrt(np.sum(x**2))/np.sqrt(np.sum(n[:,0]**2)))
    snr_out = 10 * np.log10(Px / Pn)
    # print(snr_out)
    return x+n.T, n.T

def add_snr(x,snr,K = 1):
    """ Adds noise to a signal x with SNR equal to snr.
    SNR is defined as SNR (dB) = 10 * log10(Ex/En)
    """
    N = len(x)
    x = x - np.mean(x)
    Px = np.sum(x ** 2)

    n = np.random.rand(N)
    n = n - np.mean(n)
    Pn = np.sum(n ** 2)
    n = n / np.sqrt(Pn)

    Pn = Px * 10 ** (- snr / 10)
    n = n * np.sqrt(Pn)
    # snr_out1 = 20 * np.log10(np.sqrt(np.sum(x**2))/np.sqrt(np.sum(n**2)))
    snr_out = 10 * np.log10(Px / Pn)
    # print(snr_out)
    return x+n



def empty_balls(signal, radi_seg = 1):
    Sww, stft, pos, Npad = get_spectrogram(signal)
    empty_mask = find_center_empty_balls(Sww, pos, radi_seg)
    hull_d , sub_empty= get_convex_hull(Sww, pos, empty_mask) 
    mask, xr, t, aux = reconstruct_signal_2(sub_empty, stft, Npad)
    return xr


