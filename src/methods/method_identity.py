from methods.MethodTemplate import MethodTemplate
import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import *

class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'identity'
        self.task = 'denoising'


    def method(self, signals, params):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = 2*signals
   
        return signals_output
