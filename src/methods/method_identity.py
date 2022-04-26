""" A template file to add new method in the benchmark.


Remark: Make sure that this file starts with "method_".
"""

""" First section ----------------------------------------------------------------------
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
from methods.MethodTemplate import MethodTemplate # Import the template!
import numpy as np

""" Second section ---------------------------------------------------------------------
| Put here all the functions that your method uses.
| 
| def a_function_of_my_method(signal,params):
|   ...
"""
def a_function_of_my_method(signal,params):
    return signal

""" Third section ----------------------------------------------------------------------
| Create here a new class that will encapsulate your method.
| This class should inherit the abstract class MethodTemplate.
| You must then implement the class function: 

def method(self, signal, params)

| which should receive the signals and any parameters that you desire to pass to your
| method.
"""

class NewMethod(MethodTemplate):
    def __init__(self):
        self.id = 'identity_method'
        self.task = 'denoising'  # Should be either 'denoising' or 'detection'

    def method(self, signals, params = None): # Implement this method.
        signals_out = np.zeros_like(signals)
        for k,signal in enumerate(signals):
            signals_out[k] = a_function_of_my_method(signal,params)
        
        return signals_out

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return [None,]