from benchmark_demo.benchmark_utils import MethodTemplate, MatlabInterface
# import sys
# sys.path.append("methods")
# You must import the MethodTemplate abstract class and the MatlabInterface class.

# Create an interface with the matlab engine by passing the name of the function file 
# (without the .m extension). Then get the matlab function as:

# Paths to additional code for the method to add to Matlab path variable.
paths = ['src\methods\ssa_decomp_utils',
        '..\src\methods\ssa_decomp_utils'
        ]

mlint = MatlabInterface('ssa_denoising', add2path=paths) 
matlab_function = mlint.matlab_function # A python function handler to the method.

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'ssa_denoising'
        self.task = 'denoising'
        
    def method(self, signal, *params, **kw_params):
        """_summary_

        Args:
            signals (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """

        signal_output = matlab_function(signal, *params) # Only positional args.
        return signal_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return (((25,),{}),)    
        
