""" This file contains a number of utilities for time-frequency analysis. 
Some functions has been modified from the supplementary code of:
Bardenet, R., Flamant, J., & Chainais, P. (2020). "On the zeros of the spectrogram of 
white noise." Applied and Computational Harmonic Analysis, 48(2), 682-705.

Those functions are:
- getSpectrogram(signal)
- findCenterEmptyBalls(Sww, pos_exp, radi_seg=1)
- getConvexHull(Sww, pos_exp, empty_mask, radi_expand=0.5)
- reconstructionSignal(hull_d, stft)
"""

import numpy as np
import scipy.signal as sg
from scipy.fft import fft, ifft
from math import factorial
from numpy import complex128, dtype, pi as pi
def get_gauss_window(Nfft,L,prec=1e-6):
    l=np.floor(np.sqrt(-Nfft*np.log(prec)/pi))+1
    N = 2*l+1
    t0 = l+1
    tmt0=np.arange(0,N)-t0
    g = np.exp(-(tmt0/L)**2 * pi)    
    g=g/np.linalg.norm(g)
    return g

def get_round_window(Nfft, prec = 1e-6):
    """ Generates a round Gaussian window, i.e. same essential support in time and 
    frequency: g(n) = exp(-pi*(n/T)^2) for computing the Short-Time Fourier Transform.
    
    Args:
        Nfft: Number of samples of the desired fft.

    Returns:
        g (ndarray): A round Gaussian window.
        T (float): The scale of the Gaussian window (T = sqrt(Nfft))
    """
    # analysis window
    L=np.sqrt(Nfft)
    l=np.floor(np.sqrt(-Nfft*np.log(prec)/pi))+1
 
    N = 2*l+1
    t0 = l+1
    tmt0=np.arange(0,N)-t0
    g = np.exp(-(tmt0/L)**2 * pi)    
    g=g/np.linalg.norm(g)
    return g, L

def get_stft(x,window=None,t=None,Nfft=None):
    xrow = len(x)

    if t is None:
        t = np.arange(0,xrow)

    if Nfft is None:
        Nfft = 2*xrow

    if window is None:
        window = get_round_window(Nfft)

    tcol = len(t)
    hlength=np.floor(Nfft/4)
    hlength=int(hlength+1-np.remainder(hlength,2))

    hrow=len(window)
    
    assert np.remainder(hrow,2) == 1

    Lh=(hrow-1)//2
    tfr= np.zeros((Nfft,tcol))   
    for icol in range(0,tcol):
        ti= t[icol]; 
        tau=np.arange(-np.min([np.round(Nfft/2),Lh,ti]),np.min([np.round(Nfft/2),Lh,xrow-ti])).astype(int)
        indices= np.remainder(Nfft+tau,Nfft).astype(int); 
        tfr[indices,icol]=x[ti+tau]*np.conj(window[Lh+tau])/np.linalg.norm(window[Lh+tau])
    
    tfr=fft(tfr, axis=0) 
    return tfr

def get_istft(tfr,window=None,t=None):
    
    N,NbPoints = tfr.shape
    tcol = len(t)
    hrow = len(window) 
    Lh=(hrow-1)//2
    window=window/np.linalg.norm(window)
    tfr=ifft(tfr,axis=0)

    x=np.zeros((tcol,),dtype=complex)
    for icol in range(0,tcol):
        valuestj=np.arange(np.max([1,icol-N/2,icol-Lh]),np.min([tcol,icol+N/2,icol+Lh])).astype(int)
        for tj in valuestj:
            tau=icol-tj 
            indices= np.remainder(N+tau,N).astype(int)
            x[icol]=x[icol]+tfr[indices,tj]*window[Lh+tau]
        
        x[icol]=x[icol]/np.sum(np.abs(window[Lh+icol-valuestj])**2)
    return x


# def get_stft(signal, window = None, overlap = None):
#     """ Compute the STFT of the signal. Signal is padded with zeros.
#     The outputs corresponds to the STFT with the regular size and also the
#     zero padded version. The signal is zero padded to alleviate border effects.

#     Args:
#         signal (ndarray): The signal to analyse.
#         window (ndarray, optional): The window to use. If None, uses a rounded Gaussian
#         window. Defaults to None.

#     Returns:
#         stft(ndarray): Returns de stft of the signal.
#         stft_padded(ndarray): Returns the stft of the zero-padded signal.
#         Npad(int): Number of zeros padded on each side of the signal.
#     """
    
#     N = np.max(signal.shape)
#     if window is None:
#         window, _ = get_round_window(N)

#     Npad = N//2 #0 #int(np.sqrt(N))
#     Nfft = len(window)
    
#     if overlap is None:
#         overlap = Nfft-1
    
#     if signal.dtype == complex128:
#         signal_pad = np.zeros(N+2*Npad, dtype=complex128)
#     else:
#         signal_pad = np.zeros(N+2*Npad)

#     # signal_pad = np.zeros(N+2*Npad)
#     signal_pad[Npad:Npad+N] = signal

#     # computing STFT
#     _, _, stft_padded = sg.stft(signal_pad, window=window, nperseg=Nfft, noverlap = overlap)
    
#     # if signal.dtype == complex128:
#     #     stft_padded = stft_padded[0:Nfft//2+1,:]
        
#     stft = stft_padded[:,Npad:Npad+N]
#     return stft, stft_padded, Npad


def get_spectrogram(signal,window=None,Nfft=None,t=None,onesided=True):
    """
    Get the round spectrogram of the signal computed with a given window. 
    
    Args:
        signal(ndarray): A vector with the signal to analyse.

    Returns:
        S(ndarray): Spectrogram of the signal.
        stft: Short-time Fourier transform of the signal.
        stft_padded: Short-time Fourier transform of the padded signal.
        Npad: Number of zeros added in the zero-padding process.
    """

    N = np.max(signal.shape)
    if Nfft is None:
        Nfft = 2*N

    if window is None:
        window, _ = get_round_window(Nfft)

    if t is None:
        t = np.arange(0,N)
        
    stft=get_stft(signal,window=window,t=t,Nfft=Nfft)

    if onesided:
        S = np.abs(stft[0:Nfft//2+1,:])**2
    else:
        S = np.abs(stft)**2                
    return S, stft


def find_zeros_of_spectrogram(S):
    aux_S = np.zeros((S.shape[0]+2,S.shape[1]+2))+np.Inf
    aux_S[1:-1,1:-1] = S
    S = aux_S
    aux_ceros = ((S <= np.roll(S,  1, 0)) &
            (S <= np.roll(S, -1, 0)) &
            (S <= np.roll(S,  1, 1)) &
            (S <= np.roll(S, -1, 1)) &
            (S <= np.roll(S, [-1, -1], [0,1])) &
            (S <= np.roll(S, [1, 1], [0,1])) &
            (S <= np.roll(S, [-1, 1], [0,1])) &
            (S <= np.roll(S, [1, -1], [0,1])) 
            )
    [y, x] = np.where(aux_ceros==True)
    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = y-1
    pos[:, 1] = x-1
    return pos


def reconstruct_signal(hull_d, stft):
    """Reconstruction using the convex hull.
    This function is deprecated and conserved for retrocompatibility purposes only.

    Args:
        hull_d (_type_): _description_
        stft (_type_): _description_

    Returns:
        _type_: _description_
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



def reconstruct_signal_2(mask, stft, Npad, Nfft=None, window=None, overlap=None):
    """Reconstruction using a mask given as parameter

    Args:
        mask (_type_): _description_
        stft (_type_): _description_
        Npad (_type_): _description_
        Nfft (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    Ni = mask.shape[1]
    if Nfft is None:
        Nfft = Ni
    
    if window is None:
        window = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
        window = window/window.sum()
    
    Nfft = len(window)
    
    if overlap is None:
        overlap=Nfft-1


    # reconstruction        
    mask_aux = np.zeros(stft.shape)
    mask_aux[:,Npad:Npad+Ni] = mask
    # t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask_aux*stft, window=window, nperseg=Nfft, noverlap = overlap)
    xr = xr[Npad:Npad+Ni]
    return xr, t

def reconstruct_signal_3(mask, stft, window=None):
    mask_complete = np.zeros_like(stft, dtype=bool)
    mask_complete[0:mask.shape[0],:] = mask
    mask_complete[mask.shape[0]::,:] = mask[-2:0:-1,:]
    xr = get_istft(stft*mask_complete,window=window,t=np.arange(0,stft.shape[1]))
    return xr


# def find_zeros_of_spectrogram(M,th=1e-14):
#     """ Finds the local minima of the spectrogram matrix M.

#     Args:
#         M (_type_): Matrix with real values.
#         th (_type_): A given threshold.

#     Returns:
#         _type_: _description_
#     """

#     C,R = M.shape
#     Mid_Mid = np.zeros((C,R), dtype=bool)
#     for c in range(1, C-1):
#         for r in range(1, R-1):
#             T = M[c-1:c+2,r-1:r+2]
#             Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
#             #Mid_Mid[c, r] = (np.min(T) == T[1, 1])
#     x, y = np.where(Mid_Mid)
#     pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
#     pos[:, 0] = x
#     pos[:, 1] = y
#     return pos


def snr_comparison(x,x_hat):
    """_summary_

    Args:
        x (_type_): _description_
        x_hat (_type_): _description_

    Returns:
        _type_: _description_
    """

    qrf = 10*np.log10(np.sum(x**2)/np.sum((x-x_hat)**2))
    return qrf


def add_snr_block(x,snr,K = 1):
    """Adds noise to a signal x with SNR equal to snr.
    SNR is defined as SNR (dB) = 10 * log10(Ex/En), where Ex and En are the energy of 
    the signal and the noise, respectively.

    Args:
        x (ndarray): Signal.
        snr (_type_): Signal-to-Noise Ratio in dB.
        K (int, optional): The number of noisy signals generated, vertically stacked. 
        Defaults to 1.

    Returns:
        ndarray: Block of shape [K,N], where K is the number of noisy signals, and N is 
        length of the signal x.
    """

    N = len(x)
    x = x - np.mean(x)
    Px = np.sum(x ** 2)
    # print(x)

    n = np.random.randn(N,K)
    # n = n - np.mean(n,axis = 0)
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
    

def add_snr(x,snr,complex_noise=False):
    """Adds noise to a signal x with SNR equal to snr. SNR is defined as 
    SNR (dB) = 10 * log10(Ex/En), where Ex and En are the energy of the signal and the 
    noise, respectively.

    Args:
        x (ndarray): Signal.
        snr (_type_): Signal-to-Noise Ratio in dB.
        K (int, optional): The number of noisy signals generated, vertically stacked. 
        Defaults to 1.

    Returns:
        ndarray: Block of shape [K,N], where K is the number of noisy signals, and N is 
        length of the signal x.
    """
    
    N = len(x)
    x = x - np.mean(x)
    Px = np.sum(x ** 2)

    # Create the noise for signal with given SNR:
    n = np.random.randn(N,)
    if complex_noise:
        n = n.astype(complex)
        n += 1j*np.random.randn(N,)
        
    Pn = np.sum(np.abs(n) ** 2, axis = 0) # Normalize to 1 the variance of noise.
    n = n / np.sqrt(Pn)
    Pn = Px * 10 ** (- snr / 10) # Give noise the prescribed variance.
    n = n * np.sqrt(Pn)

    # Pn = Px * 10 ** (- snr / 10)
    # n = n * np.sqrt(Pn)
    # snr_out1 = 20 * np.log10(np.sqrt(np.sum(x**2))/np.sqrt(np.sum(n**2)))
    snr_out = 10 * np.log10(Px / Pn)
    print('snr_out:{}'.format(snr_out))
    return x+n,n


def hermite_poly(t,n, return_all = False):
    """Generates a Hermite polynomial of order n on the vector t.

    Args:
        t (ndarray, float): A real valued vector on which compute the Hermite 
        polynomials.
        n (int): Order of the Hermite polynomial.

    Returns:
        ndarray: Returns an array with the Hermite polynomial computed on t.
    """

    all_hp = np.zeros((n+1,len(t)))

    if n == 0:
        hp = np.ones((len(t),))
        all_hp[0] = hp

    else:
            hp = 2*t
            all_hp[0,:] = np.ones((len(t),))
            all_hp[1,:] = hp
            # if n >= 1:
            for i in range(2,n+1):
                # hp = 2*t*hermite_poly(t,n-1) - 2*(n-1)*hermite_poly(t,n-2)
                hp = 2*t*all_hp[i-1] - 2*(i-1)*all_hp[i-2]
                all_hp[i,:] = hp

    if not return_all:
        return hp
    else:
        return hp, all_hp


def hermite_fun(N, q, t=None, T=None, return_all = False):
    """Computes an Hermite function of order q, that consist in a centered Hermite 
    polynomial multiplied by the squared-root of a centered Gaussian given by: 
    exp(-pi(t/T)^2). The parameter T fixes the width of the Gaussian function.

    Args:
        N (int): Length of the function in samples
        q (int): Order of the Hermite polynomial.
        t (ndarray): Values on which compute the function. If None, uses a centered 
        vector from -N//2 to N//2-1. Defaults to None.
        T (float): Scale of the Gaussian involved in the Hermite function. If None,
        N = sqrt(N). Defaults to None.

    Returns:
        _type_: _description_
    """

    if t is None:
        t = np.arange(N)-N//2

    if T is None:
        T = np.sqrt(N)

    _, all_hp = hermite_poly(np.sqrt(2*pi)*t/T, q-1,return_all=True)
    gaussian_basic = np.exp(-pi*(t/T)**2)/np.sqrt(T)
    hfunc = np.zeros((q,len(t)))

    for i in range(q):
        Cnorm = np.sqrt(factorial(q-1)*(2**(q-1-0.5)))
        # gaussian_basic /= np.sum(gaussian_basic)
        hfunc[i] = gaussian_basic*all_hp[i]/Cnorm
    
    if not return_all:
        return hfunc[-1]
    else:
        return hfunc[-1], hfunc
        
def sigmerge(x1, noise, ratio, return_noise=False):
    # Get signal parameters.
    N = len(x1)
    ex1=np.mean(np.abs(x1)**2)
    ex2=np.mean(np.abs(noise)**2)
    h=np.sqrt(ex1/(ex2*10**(ratio/10)))
    scaled_noise = noise*h
    sig=x1+scaled_noise

    if return_noise:
        return sig, scaled_noise
    else:
        return sig