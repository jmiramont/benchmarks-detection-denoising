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
        self.id = 'global_rank_env_test'
        self.task = 'detection'

        # A shared class attribute
        # print('Creating Compute Statistics object...')
        # self.sc = ComputeStatistics()
        # print('Finished.')

    def method(self, signal, *args, **kwargs): # Implement this method.
        reject_H0 = compute_rank_envelope_test(signal, *args, **kwargs)
        return reject_H0
          
    def get_parameters(self):            # Use it to parametrize your method.
        return [
                {'fun':'Fest','correction':'rs'},
                {'fun':'Fest','correction':'rs','rmin':0.65,'rmax':1.05},
                {'fun':'Fest','correction':'rs','transform':'asin(sqrt(.))'},
                {'fun':'Fest','correction':'rs','rmin':0.65,'rmax':1.05,
                'transform':'asin(sqrt(.))'},
                ]