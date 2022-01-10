import numpy as np
import scipy.stats as spst
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import scipy.signal as sg
import utilstf as utils


N = 256
signal = np.random.randn(N)
Nfft = N
base = int(np.sqrt(Nfft))
sig_aux = np.zeros((2*base+N))
sig_aux[base:N+base] = signal

# window
g = sg.gaussian(Nfft-1, np.sqrt((Nfft)/2/np.pi))
g = g/g.sum()
_, _, stft = sg.stft(sig_aux, window=g, nperseg=Nfft-1, noverlap = Nfft-2)
Sww1 = np.abs(stft)**2

_, _, stft = sg.stft(signal, window=g, nperseg=Nfft-1, noverlap = Nfft-2)
Sww2 = np.abs(stft)**2


fig,axs = plt.subplots(1,2)
axs[0].imshow(Sww1, origin = 'lower')
axs[1].imshow(Sww2, origin = 'lower')

plt.show()