# One possible metric of musical noise:
import librosa
import numpy as np
import scipy.signal as sg
from numpy.linalg import norm
import os

# Let's define a performance function from Matlab
from mcsm_benchmarks.benchmark_utils import MatlabInterface


def tkeo(x):
    m=1
    M=1
    y = np.zeros_like(x)
    for n in range(M,len(x)-M):
        y[n] = x[n]**(2/m) - (x[int(n-M)]*x[int(n+M)])**(1/m)
        # y[n] = x[n]*x[n+m-2]-x[n-1]*x[n+m-1]
    return y

def activity(x,fs=None):
    y = librosa.resample(x, orig_sr=fs, target_sr=48000)
    
    act = np.abs(tkeo(y))
    act = act/np.max(act)
    thr = np.quantile(act,0.9)
    act[act>thr] = 1
    act[act<=thr] = 0
    act = act.astype(bool)
    act = np.invert(act)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1,1)
    # axs.plot(y)
    # axs.plot(act*np.max(np.abs(y)))
    # axs.plot(np.ones_like(act)*thr)
    # plt.show()

    return act

def activity_2(x,fs=None,q=None):

    y = librosa.resample(x, orig_sr=fs, target_sr=48000)
    
    b, a = sg.butter(2, 0.05, btype='lowpass', output='ba')
    env = sg.filtfilt(b,a, np.abs(y)**2)
    env = np.sqrt(2*np.abs(env))
    act = env.copy()

    thr = np.quantile(act,q) #np.mean(act**2)**0.5
    act[act>thr] = 1
    act[act<=thr] = 0
    act = act.astype(bool)
    act = np.invert(act)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1,1)
    # axs.plot(y)
    # axs.plot(env)
    # axs.plot(act*np.max(np.abs(y)))
    # axs.plot(np.ones_like(act)*thr)
    # plt.show()

    return act

# Spectral Kurtosis
def spectral_kurt(X):
    eps=1e-15
    bar_X = np.mean(X,axis=0)
    diff = X-bar_X
    numerator = np.mean(diff**4,axis=0)
    denominator= np.mean(diff**2,axis=0)**2
    kurtX = numerator/(denominator+eps)
    return kurtX

# # (1) Musical Noise measurement: log kurtosis ratio
# def log_kurt_ratio(xi,xo):
#     Xi = np.abs(librosa.stft(xi))
#     Xo = np.abs(librosa.stft(xo))
#     kurtXi = np.mean(spectral_kurt(Xi))
#     kurtXo = np.mean(spectral_kurt(Xo))
#     DeltaK = np.log(kurtXo/kurtXi)
#     if DeltaK < 0:
#         DeltaK = 0

#     if DeltaK > 1.4:
#         DeltaK = 1.4
    
#     DeltaK = 100*(1-DeltaK/1.4)

#     return DeltaK 

# (2) Musical Noise measurement: Perceptually Improved Log-Kurtosis Ratio
def compute_Xplus(x,fs,pthr=0.05):
    eps = 1e-15
    wl = 1024
    stft = librosa.stft(x, window='cosine', n_fft=2*wl, win_length=wl, hop_length=wl//2)
    X_dBA = np.abs(stft)**2
    f = np.linspace(0,fs/2,X_dBA.shape[0])

    # Get the dBA
    Aw = 10**(librosa.A_weighting(f)/10)

    for col in range(X_dBA.shape[1]):
        X_dBA[:,col] = X_dBA[:,col] * Aw
    
    PdBA = 10*np.log10(np.mean(np.mean(X_dBA,axis=0))+eps)
    X_dBA = 10*np.log10(X_dBA+eps)    

    thr = PdBA-20
    Xplus = X_dBA-thr
    Xplus[Xplus<0] = 0
    
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(X_dBA, origin='lower', aspect='auto')
    # axs[1].imshow(Xplus, origin='lower', aspect='auto')
    # plt.show()
    
    # Check frames where the energy is 0 for all frequencies to remove them later.
    ind_frames = np.sum(Xplus, axis=0) > 0
    
    # Get sub-bands:
    Xplus_subb = list()

    # Band 1: Between 50Hz and 750Hz
    fb1 = np.logical_and(f > 50, f <= 750)
    Xplus_subb.append(Xplus[fb1,:])

    # Band 2: Between 50Hz and 750Hz
    fb2 = np.logical_and(f > 750, f <= 6000)
    Xplus_subb.append(Xplus[fb2,:])

    # Band 3: Between 50Hz and 750Hz
    fb3 = np.logical_and(f > 6000, f <= np.min([f[-1],16000]))
    Xplus_subb.append(Xplus[fb3,:])

    return Xplus_subb, ind_frames

def perceptual_kurt_ratio(xi,xo,fs,act=None):
    # Resample signals:
    xi_rs = librosa.resample(xi, orig_sr=fs, target_sr=48000)
    xo_rs = librosa.resample(xo, orig_sr=fs, target_sr=48000)

    xi_rs = xi_rs*act
    xo_rs = xo_rs*act
    xi_rs = xi_rs[act>0]
    xo_rs = xo_rs[act>0]

    Xplus_subb_i,idx_i = compute_Xplus(xi_rs,48000)
    Xplus_subb_o,idx_o = compute_Xplus(xo_rs,48000)

    idx = np.logical_and(idx_i,idx_o)
    Xplus_subb_i = [X[:,idx_i] for X in Xplus_subb_i]
    Xplus_subb_o = [X[:,idx_i] for X in Xplus_subb_o]
    
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(Xplus_subb_i[2], origin='lower', aspect='auto')
    # axs[1].imshow(Xplus_subb_o[2], origin='lower', aspect='auto')
    # plt.show()

    # Spectral Weights
    # wi = [10*np.log10(np.mean(10**(X/10),axis=0)) for X in Xplus_subb_i]
    wo = [10*np.log10(np.mean(10**(X/10),axis=0)) for X in Xplus_subb_o]

    # Spectral Kurtosis of the subbands
    kurt_subb_i = [spectral_kurt(X) for X in Xplus_subb_i]
    kurt_subb_o = [spectral_kurt(X) for X in Xplus_subb_o]

    # Spectral log-kurtosis ratios
    eps = 1e-15
    DeltaK_B = []
    for i in range(len(kurt_subb_o)):
        cociente = kurt_subb_o[i]/(kurt_subb_i[i]+eps)
        temp = np.abs(np.log(cociente+eps))
        # DeltaK_B.append([np.min([q,0.5]) for q in temp])
        DeltaK_B.append(temp)

    DeltaK_PI_aux = [np.sum(w*DeltaK)/(np.sum(w)+eps) for w, DeltaK in zip(wo,DeltaK_B)]
    # DeltaK_PI_aux = [np.min([tmp,0.5]) for tmp in DeltaK_PI_aux]
    band_index = np.nanargmax(DeltaK_PI_aux)
    # DeltaK_B = DeltaK_PI_aux[band_index]
    lb = 0.5
    DeltaK_B = np.min([DeltaK_PI_aux[band_index],lb])
    DeltaK_PI = 100*(1-DeltaK_B/lb)
    return DeltaK_PI 

def musical_noise_measure(x,x_hat,*args,**kwargs):
        DeltaK_PI = np.zeros((x_hat.shape[0],))
        if 'scaled_noise' in kwargs.keys():
            xn = kwargs['scaled_noise']

        fs = x._fs
        # Compute activity
        y = sg.resample(x,int(len(x)*48000/fs))
        y = y / np.max(np.abs(y))
        act = activity(y)
        act = np.invert(act)    

        for i,xo in enumerate(x_hat):
            DeltaK_PI[i] = perceptual_kurt_ratio(x+xn[i],xo,fs=fs,act=act) 

        return DeltaK_PI

def aps_measure(x,xn,x_hat,*args,**kwargs):
    # APS from PEASS matlab implementation
    paths = [os.path.join('src','peass'),
            os.path.join('..','src','peass')
            ]
    
    mlint = MatlabInterface('APS_wrapper', add2path=paths) 
    aps_wrapper = mlint.matlab_function # A python function handler to the method.
    fs = 11025
    aps = aps_wrapper(x,xn,x_hat,fs)
    
    return aps

def musical_noise_measure_aps(x,x_hat,*args,**kwargs):
        if 'scaled_noise' in kwargs.keys():
            xn = kwargs['scaled_noise']

        APS = np.zeros((x_hat.shape[0],))
        for i,_ in enumerate(x_hat):
            APS[i] = aps_measure(x,xn[i],x_hat[i]) 

        return APS


