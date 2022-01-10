import numpy as np
from matplotlib import pyplot as plt


def add_snr(x,snr,K = 1):
    """
    Adds noise to a signal x with SNR equal to snr. SNR is defined as SNR (dB) = 10 * log10(Ex/En)
    """
    N = len(x)
    x = x - np.mean(x)
    Px = np.sum(x ** 2)
    # print(x)

    n = np.random.rand(N,K)
    n = n - np.mean(n,axis = 0)
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
    return x+n.T

N = 128
x = np.sin(2*np.pi*np.arange(N)/N)

xn, n = add_snr(x,-10,1)

print(np.mean(xn, axis = 1))

fig, axs = plt.subplots(5,1)
for k, ax in enumerate(axs):
    ax.plot(xn[k])


plt.show()

