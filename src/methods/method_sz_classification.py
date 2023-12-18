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

paths = [   os.path.join('src','methods','sz_classification_utils'),
            os.path.join('..','src','methods','sz_classification_utils')
        ]

mlint = MatlabInterface('szc_method', add2path=paths) 

matlab_function = mlint.matlab_function # A python function handler to the method.

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'sz_classification'
        self.task = 'denoising'
        

    def method(self, signal, *params):
        """_summary_

        Args:
            signals (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        # szc_method(signal,noise_std,J,criterion,margins,beta,Nfft)
        signal_output = matlab_function(signal, *params) # Only positional args.
        return signal_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return (((25,),{}),)    
# xr = brevdo_method(x, Ncomp, use_sst, Pnei)        
