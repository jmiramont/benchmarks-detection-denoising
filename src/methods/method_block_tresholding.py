from methods.benchmark_utils import MethodTemplate, MatlabInterface
# You must import the MethodTemplate abstract class and the MatlabInterface class.

# Create an interface with the matlab engine by passing the name of the function file 
# (without the .m extension). Then get the matlab function as:
mlint = MatlabInterface('BlockThresholding') 
matlab_function = mlint.matlab_function # A python function handler to the method.

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'block_thresholding'
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

    def get_parameters(self):            # Use it to parametrize your method.
        return (((25,),{}),)    
        
