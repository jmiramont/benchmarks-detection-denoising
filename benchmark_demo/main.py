import numpy as np
from numpy import pi as pi
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from utilstf import *

np.random.seed(12) 
# signal parameters
SNR = 50
N = 2**7
t = np.arange(N)/N
tmin = int(np.sqrt(N))
tmax = N-int(np.sqrt(N))
g0 = 1e-3# g0 = g(0) = g(1)
sigma = -np.log(g0)*4
g = np.exp(-sigma*(t-0.5)**2)
# s = g*np.cos(2*pi*(N/8*t+N/8*t*t))
phi = N/3*t + N/9 * np.sin(2*pi*t)/2/pi
s = np.zeros((2*tmin+N,))
s[tmin:tmin+N] = np.cos(2*pi*phi)
s[tmin:tmin+N] *= g
signal = add_snr(s,SNR)

pos, [Sww, stft, x, y] = getSpectrogram(signal)
empty_mask = findCenterEmptyBalls(Sww, pos, radi_seg=1)
hull_d , sub_empty= getConvexHull(Sww, pos, empty_mask)
mask, xr, t, aux = reconstructionSignal2(sub_empty, stft)

print(Sww.shape)
fig, ax = plt.subplots(1,2,figsize = (10,5))
ax[0].imshow(np.log10(Sww), origin='lower', cmap=cmocean.cm.deep)
ax[1].imshow(np.abs(aux), origin='lower')
# plt.imshow(empty_mask, origin='lower')
# plt.scatter(x, y, color='w', s=40)

plt.figure()
plt.plot(s)
plt.plot(xr,'--')
plt.show()
