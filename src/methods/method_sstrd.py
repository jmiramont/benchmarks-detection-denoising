from mcsm_benchs.benchmark_utils import MethodTemplate
from mcsm_benchs.MatlabInterface import MatlabInterface
import os
# import sys
# sys.path.append("methods")
# You must import the MethodTemplate abstract class and the MatlabInterface class.

# Create an interface with the matlab engine by passing the name of the function file 
# (without the .m extension). Then get the matlab function as:

# Paths to additional code for the method to add to Matlab path variable.
# paths = ['src\methods\dbrevdo_methods_utils',
        # '..\src\methods\dbrevdo_methods_utils'
        # ]

paths = [   os.path.join('src','methods','sstrd_method_utils'),
            os.path.join('..','src','methods','sstrd_method_utils')
        ]

mlint = MatlabInterface('sstrd_method', add2path=paths) 

matlab_function = mlint.matlab_function # A python function handler to the method.

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'sstrd_method'
        self.task = 'denoising'
        

    def method(self, signal, nc=[], *params):
        """_summary_

        Args:
            signals (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        if nc == []:
            nc = signal.total_comps
     
        signal_output = matlab_function(signal, nc, *params) # Only positional args.
        return signal_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return (((25,),{}),)    
# xr = brevdo_method(x, Ncomp, use_sst, Pnei)        
