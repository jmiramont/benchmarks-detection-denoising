import numpy as np
from numpy import pi as pi
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from utilstf import *

np.random.seed(0) 
# signal parameters
SNR = 10
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

sr = emptyBalls(signal, radi_seg = 0.9)

print(snrComparison(s,sr))

# plt.figure()
# plt.plot(s)
# plt.plot(sr,'--')
# plt.show()