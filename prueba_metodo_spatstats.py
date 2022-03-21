
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from methods.spatstats_utils import ComputeStatistics, compute_hyp_test
from math import atan2
from benchmark_demo.SignalBank import SignalBank
from spatstat_interface.interface import SpatstatInterface

# import py_compile
# py_compile.compile('src/benchmark_demo/finding_zeros.py','src/benchmark_demo/finding_zeros.pyc')

# Parameters
N = 2**8
Nfft = N
SNRin = 5
statistics = ('L','Frs','Fcs','Fkm')
pnorm = 2
reps = 1
np.random.seed(0)


sbank = SignalBank(N = N, Nsub=128)
chirp = sbank.signal_linear_chirp()
# chirp = sbank.signal_mc_harmonic()
signal = add_snr(chirp,SNRin)
g,_ = get_round_window(Nfft)
S, stft, stft_padded, Npad = get_spectrogram(chirp, window = g)
# plt.figure()
# plt.imshow(S, origin='lower')
# plt.show()

SNRs = (5,)# 10, 15, 20)
radius = np.arange(0.0, 4.0, 0.01)
# radius = np.linspace(0.0, 3.0)

print('Instantiating SpatstatInterface...')
spatstat = SpatstatInterface(update=False)  
sc = ComputeStatistics(spatstat=spatstat)
# output = np.zeros((reps,len(radius)))
for SNRin in SNRs:
    output = {sts:list() for sts in statistics}
    signal = add_snr(chirp,SNRin)
    for j in range(reps):
        print(j)
        hyp_test_dict = compute_hyp_test(signal, sc=sc, MC_reps = 199, alpha = 0.05,
                                    statistic=statistics, pnorm = pnorm,
                                    radius = radius)
        for sts in statistics:
            output[sts].append(hyp_test_dict[sts])
        

    # for sts in statistics:
        # np.save('outputmat_{}_N_{}_SNRin_{}_{}.npy'.format(sts,N,SNRin,pnorm), output[sts])


print('Script finished.')