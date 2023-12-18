""" A template file to add new method in the benchmark.


Remark: Make sure that this file starts with "method_".
"""

""" First section ----------------------------------------------------------------------
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
from mcsm_benchs.benchmark_utils import MethodTemplate
from mcsm_benchs.MatlabInterface import MatlabInterface


""" Second section ---------------------------------------------------------------------
| After moving a file called 'my_matlab_method.m' to 
| src\methods, create an interface with the matlab engine by 
| passing the name of the function file (without the .m 
| extension). Then get the matlab function as:
"""
mlint = MatlabInterface('my_matlab_method') 
matlab_function = mlint.matlab_function # A python function handler to the method.



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
        self.id = 'my_matlab_method'
        self.task = 'denoising'
        

    def method(self, signal, *params): # Only positional args for matlab methods.
        signal_output = matlab_function(signal, *params) # Only positional args.
        return signal_output

    # def get_parameters(self):            # Use this to parametrize your method.
    #     return [None,]