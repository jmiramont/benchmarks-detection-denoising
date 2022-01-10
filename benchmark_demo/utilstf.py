import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sg
from math import atan2
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay


"""This file contains a number of utilities for time-frequency analysis. Some functions has been modified from the supplementary code of:
    Bardenet, R., Flamant, J., & Chainais, P. (2020). On the zeros of the spectrogram of white noise.
    Applied and Computational Harmonic Analysis, 48(2), 682-705.
which can be found in: http://github.com/jflamant/2018-zeros-spectrogram-white-noise.
  
Those functions are:
- getSpectrogram(signal)
- findCenterEmptyBalls(Sww, pos_exp, radi_seg=1)
- getConvexHull(Sww, pos_exp, empty_mask, radi_expand=0.5)
- reconstructionSignal(hull_d, stft)



"""

def getSpectrogram(signal):
    """
    Get the round spectrogram of the signal 
    """
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
    Sww = np.abs(stft[:,Npad:Npad+Ni])**2
    # tmin = base
    # tmax = base+N
    # Sww = Sww_t[:, tmin:tmax]
    # stft = stft[:, tmin:tmax]

    # detection of zeros of the spectrogram
    th = 1e-14
    y, x = extr2minth(Sww, th) # Find zero's coordinates

    u = (np.array(x))/np.sqrt(Nfft) # Normalize the coordinates.
    v = (np.array(y))/np.sqrt(Nfft)

    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = u
    pos[:, 1] = v

    return Sww, stft, pos, Npad #[pos, x, y]


def findCenterEmptyBalls(Sww, pos_exp, radi_seg=1):
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


def getConvexHull(Sww, pos_exp, empty_mask, radi_expand=0.5):
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
    
    # plot
    # hull = ConvexHull(points)
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.imshow(np.log10(Sww), origin='lower', cmap=cmocean.cm.deep)
    # ax.scatter(pos_exp[:, 0]*np.sqrt(Nfft), pos_exp[:, 1]*np.sqrt(Nfft), color='w', s=40)

    # for simplex in hull.simplices:
    #     plt.plot((points[simplex, 1]+tmin), (points[simplex, 0]+fmin), 'g-', lw=4)

    # # ax.set_xlim([tmin, tmax])
    # # ax.set_ylim([fmin, fmax])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.04, bottom=0.05)
    
    mask = np.zeros(Sww.shape)
    mask[fmin:fmax, tmin:tmax] = sub_empty
    return hull_d, mask
   


def reconstructionSignal(hull_d, stft):
    """ Recontstruction using the convex hull
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

def reconstructionSignal2(mask, stft, Npad):
    """ Reconstruction using a mask given as parameter"""
    Ni = mask.shape[1]
    Nfft = Ni
    # reconstruction
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    mask_aux = np.zeros(stft.shape)
    mask_aux[:,Npad:Npad+Ni] = mask
    # t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask_aux*stft, window=g, nperseg=Nfft, noverlap = Nfft-1)
    return xr[Npad:Npad+Ni], t


def extr2minth(M,th):
    """
    Finds the minima of the spectrogram matrix M
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


def snrComparison(x,x_hat):
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
    """
    Adds noise to a signal x with SNR equal to snr. SNR is defined as SNR (dB) = 10 * log10(Ex/En)
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


def hardThresholding(F):
    stdNoise = np.median(np.abs(np.real(F)))/0.6745
    thr = 3*stdNoise
    mask = np.abs(F)
    mask[mask<=thr] = 0
    mask[mask>thr] = 1
    return F*mask


def emptyBalls(signal, radi_seg = 1):
    pos, [Sww, stft, x, y] = getSpectrogram(signal)
    empty_mask = findCenterEmptyBalls(Sww, pos, radi_seg)
    hull_d , sub_empty= getConvexHull(Sww, pos, empty_mask) 
    mask, xr, t, aux = reconstructionSignal2(sub_empty, stft)
    return xr


