import numpy as np
from numpy import pi as pi
import pandas as pd
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from utilstf import *
from benchmark import Benchmark
from benchmark import dic2df
import signals_bank

bank = signals_bank.SignalBank(N = 128)
s = bank.linearChirp()
# s = bank.crossedLinearChirps()
# s = bank.multiComponentHarmonic(a1 = 16)
# s = bank.dumpedCos()
signal = add_snr(s,40)

Sww1, stft, pos, Npad  = getSpectrogram(signal)

radi_seg = 1
empty_mask = findCenterEmptyBalls(Sww1, pos, radi_seg = radi_seg)
hull_d , sub_empty= getConvexHull(Sww1, pos, empty_mask,radi_expand=radi_seg/2)
xr, t = reconstructionSignal2(sub_empty, stft, Npad)

Sww2, stft, pos, Npad  = getSpectrogram(xr)
print( 10*np.log10(np.sum(s**2)/np.sum((xr-s)**2)))
# print(Sww.shape)


fig, ax = plt.subplots(2,2,figsize = (10,5))
ax[0,0].imshow(Sww1, origin = 'lower')
ax[0,1].imshow(sub_empty, origin = 'lower')
ax[1,0].plot(signal)
ax[1,1].plot(xr)
ax[1,1].plot(s,'--')


plt.show()

