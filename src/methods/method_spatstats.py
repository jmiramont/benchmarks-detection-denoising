from methods.MethodTemplate import MethodTemplate # You must import the MethodTemplate abstract class.

"""
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
import numpy as np
from methods.spatstats_utils import *

""" 
| Put here all the functions that your method uses.
| Remark: Make sure that this file starts with "method_".
"""

""" Create here a new class that will encapsulate your method.
This class should inherit the abstract class MethodTemplate.
By doing this, you must then implement the class method: 

def method(self, signal, params)

which should receive the signals and any parameters
that you desire to pass to your method.You can use this file as an example.
"""

class NewMethod(MethodTemplate):
    
    def __init__(self):
        self.id = 'spatial_stats'
        self.task = 'detection'

        # A shared class attribute
        
        self.sc = ComputeStatistics()

    def method(self, signals, params): # Implement this method.
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        reject_H0 = np.zeros((signals.shape[0],), dtype = bool)
        for i, signal in enumerate(signals):
            print(i)
            if params is None:
                reject_H0[i] = compute_hyp_test(signal, sc=self.sc, rmax=2.0)
            else:
                reject_H0[i] = compute_hyp_test(signal, sc=self.sc, **params)

        return reject_H0
        
        
    def get_parameters(self):            # Use it to parametrize your method.
        return [{'statistic':'L', 'pnorm': 2, 'rmax': 0.5},
                {'statistic':'L', 'pnorm': 2, 'rmax': 1.0},
                {'statistic':'L', 'pnorm': 2, 'rmax': 2.0},
                # {'statistic':'L', 'pnorm': 2, 'rmax': 5.0},
                {'statistic':'Frs', 'pnorm': 2, 'rmax': 0.5},
                {'statistic':'Frs', 'pnorm': 2, 'rmax': 1.0},
                {'statistic':'Frs', 'pnorm': 2, 'rmax': 2.0},
                # {'statistic':'Frs', 'pnorm': 2, 'rmax': 5.0},
                {'statistic':'L', 'pnorm': np.inf, 'rmax': 0.5},
                {'statistic':'L', 'pnorm': np.inf, 'rmax': 1.0},
                {'statistic':'L', 'pnorm': np.inf, 'rmax': 2.0},
                # {'statistic':'L', 'pnorm': np.inf, 'rmax': 5.0},
                {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 0.5},
                {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 1.0},
                {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 2.0},]
                # {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 5.0},]