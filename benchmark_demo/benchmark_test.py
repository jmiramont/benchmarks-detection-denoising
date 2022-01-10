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


def a_method(noisy_signals, params):
    # results = np.zeros(noisy_signals.shape)
    # for i in range(noisy_signals.shape[0]):
    #     signal = noisy_signals[i,:]
    #     pos, [Sww, stft, _, _] = getSpectrogram(signal)
    #     empty_mask = findCenterEmptyBalls(Sww, pos, radi_seg=0.9)
    #     _ , sub_empty= getConvexHull(Sww, pos, empty_mask)
    #     _, xr, _, _ = reconstructionSignal2(sub_empty, stft)
    #     results[i,0:len(xr)] = xr
    results = noisy_signals 
    return results

my_methods = {'Method 1': a_method, 'Method 2': a_method}

my_parameters = {
    'Method 1': [[3, 4, True, 'all'],[2, 1, False, 'one']],
    'Method 2': [[3, 4, True, 'all'],[2, 1, False, 'one']]
    }

my_benchmark = Benchmark(task = 'denoising', methods = my_methods, N = 256, parameters = my_parameters, SNRin = [40,50], repetitions = 3)
my_results = my_benchmark.runTest()
df = my_benchmark.getResults()

print(df)