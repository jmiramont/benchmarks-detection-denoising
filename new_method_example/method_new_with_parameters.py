""" A template file to add new method in the benchmark.


Remark: Make sure that this file starts with "method_".
"""

""" First section ----------------------------------------------------------------------
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
from mcsm_benchs.benchmark_utils import MethodTemplate # Import the template!


""" Second section ---------------------------------------------------------------------
| Put here all the functions that your method uses.
| 
| def a_function_of_my_method(signal,params):
|   ...
"""

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
        self.id = 'a_new_method'
        self.task = 'denoising'  # Should be either 'denoising' or 'detection'

    def method(self, signal, *args, **kwargs): # Implement this method.
        ...

    def get_parameters(self):            # Use it to parametrize your method.
        """ This function should return a list/tuple of positional arguments and keyword
        arguments for your method. The positional arguments must be indicated in a tuple
        or a list, whereas the keyword arguments must be indicated using a dictionary. 
        If your method uses either just positional (resp. keyword) arguments, leave an
        empty tuple (resp. dictionary).
        """
        # # Example 1. Pass a method a combination of positional/keyword args:
        # return (((5, 6),{'a':True,'b':False}),
        #         ((2, 1),{'a':False,'b':True}),     
        #         )
        # # Example 2. Use only positional args:
        # return (((5, 6),{}),
        #         ((2, 1),{}),     
        #         )
        # # Example 3. Use only keyword args:
        # return (((),{'a':True,'b':False}),
        #         ((),{'a':False,'b':True}),     
        #         )