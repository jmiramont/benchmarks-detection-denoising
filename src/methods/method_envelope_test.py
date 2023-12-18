from mcsm_benchs.benchmark_utils import MethodTemplate # You must import the MethodTemplate abstract class.

"""
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
import numpy as np
from src.utilities.spatstats_utils import *

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
        self.id = 'monte_carlo_test'
        self.task = 'detection'

        # A shared class attribute
        # print('Creating Compute Statistics object...')
        # self.sc = ComputeStatistics()
        # print('Finished.')

    def method(self, signals, *args, **kwargs): # Implement this method.
        reject_H0 = compute_envelope_test(signals, cs=None, *args, **kwargs)
        return reject_H0
        
        
    def get_parameters(self):            # Use it to parametrize your method.
        return [
                {'statistic':'Frs',     'pnorm': 2, 'rmax': 0.5, 'MC_reps' : 2499,},
                {'statistic':'Frs_vs',  'pnorm': 2, 'rmax': 0.5, 'MC_reps' : 2499,},
                {'statistic':'Frs',     'pnorm': 2, 'rmax': 1.0, 'MC_reps' : 2499,},
                {'statistic':'Frs_vs',  'pnorm': 2, 'rmax': 1.0, 'MC_reps' : 2499,},
                {'statistic':'Frs',     'pnorm': 2, 'rmax': 2.0, 'MC_reps' : 2499,},
                {'statistic':'Frs_vs',  'pnorm': 2, 'rmax': 2.0, 'MC_reps' : 2499,},
                {'statistic':'Frs',     'pnorm': 2, 'rmax': 1.0, 'MC_reps' : 199,},
                {'statistic':'Frs_vs',  'pnorm': 2, 'rmax': 1.0, 'MC_reps' : 199,},
                # {'statistic':'Frs', 'pnorm': 2, 'rmax': 5.0},
                # {'statistic':'L', 'pnorm': np.inf, 'rmax': 0.5},
                # {'statistic':'L', 'pnorm': np.inf, 'rmax': 1.0},
                # {'statistic':'L', 'pnorm': np.inf, 'rmax': 2.0},
                # {'statistic':'L', 'pnorm': np.inf, 'rmax': 5.0},
                # {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 0.5},
                # {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 1.0},
                # {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 2.0},
                # {'statistic':'Frs', 'pnorm': np.inf, 'rmax': 5.0},
                ]