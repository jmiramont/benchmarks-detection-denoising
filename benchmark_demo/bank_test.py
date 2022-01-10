import numpy as np
from numpy import pi as pi
import pandas as pd
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from zeros_benchmark.utilstf import *
from benchmark import Benchmark
from benchmark import dic2df
import signals_bank

bank = signals_bank.SignalBank(N = 128)
s = bank.linearChirp()
# s = bank.crossedLinearChirps()
# s = bank.multiComponentHarmonic(a1 = 16)
# s = bank.dumpedCos()
signal = add_snr(s,10)

pos, [Sww1, stft, x, y] = getSpectrogram(signal)
Sww2 = np.abs(hardThresholding(stft))**2

# radi_seg = 1
# empty_mask = findCenterEmptyBalls(Sww, pos, radi_seg = radi_seg)
# hull_d , sub_empty= getConvexHull(Sww, pos, empty_mask,radi_expand=radi_seg/2)
# mask, xr, t, aux = reconstructionSignal2(sub_empty, stft)
# print( 10*np.log10(np.sum(s**2)/np.sum((xr-s)**2)))
# print(Sww.shape)


fig, ax = plt.subplots(1,2,figsize = (10,5))
# ax[0].imshow(np.log10(Sww), origin='lower', cmap=cmocean.cm.deep)
ax[0].imshow(Sww1, origin = 'lower')
# ax[1].imshow(np.abs(aux), origin='lower')
ax[1].imshow(Sww2, origin = 'lower')
plt.show()

