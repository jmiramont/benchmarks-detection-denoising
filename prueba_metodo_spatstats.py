import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from methods.spatstats_utils import ComputeStatistics, compute_hyp_test
from math import atan2
from benchmark_demo.SignalBank import SignalBank

# import py_compile
# py_compile.compile('src/benchmark_demo/finding_zeros.py','src/benchmark_demo/finding_zeros.pyc')

# Graphics and plot
def label_line(line, label_text, near_i=None, near_x=None, near_y=None, rotation_offset=0, offset=(0,0)):
    """call
        l, = plt.loglog(x, y)
        label_line(l, "text", near_x=0.32)
    """
    def put_label(i, axis):
        """put label at given index"""
        i = min(i, len(x)-2)
        dx = sx[i+1] - sx[i]
        dy = sy[i+1] - sy[i]
        rotation = (np.rad2deg(atan2(dy, dx)) + rotation_offset)*0
        pos = [(x[i] + x[i+1])/2. + offset[0], (y[i] + y[i+1])/2 + offset[1]]
        axis.text(pos[0], pos[1], label_text, size=12, rotation=rotation, color = line.get_color(),
        ha="center", va="center", bbox = dict(ec='1',fc='1', alpha=1., pad=0))

    x = line.get_xdata()
    y = line.get_ydata()
    ax = line.axes
    if ax.get_xscale() == 'log':
        sx = np.log10(x)    # screen space
    else:
        sx = x
    if ax.get_yscale() == 'log':
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0: # sanitize negative i
            i = len(x) + i
        put_label(i, ax)
    elif near_x is not None:
        for i in range(len(x)-2):
            if (x[i] < near_x and x[i+1] >= near_x) or (x[i+1] < near_x and x[i] >= near_x):
                put_label(i, ax)
    elif near_y is not None:
        for i in range(len(y)-2):
            if (y[i] < near_y and y[i+1] >= near_y) or (y[i+1] < near_y and y[i] >= near_y):
                put_label(i, ax)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")

def plotRankEnvRes(radius, k, t2, t2_exp): #, tinfty, t2_exp, tinfty_exp):
    lsize=16 # labelsize

    fig, ax = plt.subplots(figsize=(4, 4))

    lk, = ax.plot(radius, t2[k, :], color='g', alpha=1)
    ax.fill_between(radius, 0, t2[k, :], color='g', alpha=.8)
    label_line(lk, r'$t_k$', near_i=49, offset=(0.3, 0))
    t2_exp.resize((t2_exp.size,))
    lexp, = ax.plot(radius, t2_exp, color='k')
    label_line(lexp, r'$t_{\mathrm{exp}}$', near_i=49, offset=(0.3, 0))

    ax.set_ylabel(r'$T_2$' + r'$\mathrm{-statistic}$', fontsize=lsize)
    ax.set_xlabel(r'$r_{\mathrm{max}}$', fontsize=lsize)
    ax.set_xlim([0, 4])
    ax.set_yticks(np.linspace(0, 0.20, 5))
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    sns.despine(offset=10)
    fig.subplots_adjust(left=.25, right=0.9, bottom=0.2, top=0.97)


# Parameters
N = 2**8
Nfft = N
SNRin = 5
statistics = ('L','Frs','Fcs','Fkm')
pnorm = 2
reps = 200
np.random.seed(0)


sbank = SignalBank(N = N, Nsub=128)
chirp = sbank.signal_linear_chirp()
# chirp = sbank.signal_mc_harmonic()
signal = add_snr(chirp,SNRin)
g,_ = get_round_window(Nfft)
S, stft, stft_padded, Npad = get_spectrogram(chirp, window = g)
plt.figure()
plt.imshow(S, origin='lower')
plt.show()

SNRs = (5, 10, 15, 20)
radius = np.arange(0.0, 4.0, 0.01)
# radius = np.linspace(0.0, 3.0)
sc = ComputeStatistics()
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
        

    for sts in statistics:
        np.save('outputmat_{}_N_{}_SNRin_{}_{}.npy'.format(sts,N,SNRin,pnorm), output[sts])


# print('a')