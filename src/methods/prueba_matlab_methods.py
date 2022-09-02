import matlab.engine
import numpy as np
from numpy import pi as pi
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from benchmark_demo.SignalBank import SignalBank
from benchmark_demo.benchmark_utils import MatlabInterface
from methods.method_block_tresholding import NewMethod
# import sys
# sys.path.append("src\methods")

np.random.seed(0)
# signal parameters
SNRin = 5
N = 2**9
Nsub = 2**8
sbank = SignalBank(N=N,Nsub=Nsub)
tmin = sbank.tmin
tmax = sbank.tmax
# s = sbank.signal_linear_chirp()
# s = sbank.signal_mc_parallel_chirps_unbalanced()
# s = sbank.signal_mc_parallel_chirps()
s = np.real(sbank.signal_mc_multi_linear())
# s = sbank.signal_cos_chirp()
# s = sbank.signal_mc_double_cos_chirp()
# s = sbank.signal_mc_on_off_tones()
# s = sbank.signal_mc_synthetic_mixture_2() # Mala.
# s = sbank.signal_mc_multi_cos_2()

signal, noise = add_snr(s,SNRin, complex_noise=False)
signal = s + noise*np.sqrt(N/Nsub) 

signal.tofile('output.csv', sep=',', format='%f')

# Start Matlab Engine
# eng = matlab.engine.start_matlab()
# eng.eval("addpath('src/methods')")
# signal2 = matlab.double(vector=signal.tolist())

# mlint = MatlabInterface('BlockThresholding')

# funa = mlint.matlab_function
# ret = funa(signal2, matlab.double(20.0), matlab.double(N), matlab.double(10**(-SNRin/20)))
methodml = NewMethod()
funa = methodml.method
b = funa(signal, 20.0, N, 10**(-SNRin/20))
# ret = eng.BlockThresholding(signal2, matlab.double(20.0), matlab.double(N), matlab.double(10**(-SNRin/20)))



# print(ret)
# plt.plot(ret)
# plt.show()

# b = np.array(ret[0].toarray())
# print(b)

S, _, _, _ = get_spectrogram(signal)
Srec, _, _, _ = get_spectrogram(b)


f, ax = plt.subplots(1,2)
ax[0].imshow((S))
ax[1].imshow((Srec))
plt.show()
