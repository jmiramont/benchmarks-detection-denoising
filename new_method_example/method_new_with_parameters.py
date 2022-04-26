""" A template file to add new method in the benchmark.


Remark: Make sure that this file starts with "method_".
"""

""" First section ----------------------------------------------------------------------
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
from methods.MethodTemplate import MethodTemplate # Import the template!


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

    def method(self, signals, params = None): # Implement this method.
        ...

    def get_parameters(self):            # Use it to parametrize your method.
         return ((1, 2),(3, 4),)

    # Other option: a list of dictionaries and **kwargs.
    # def get_parameters(self):
    #     return [{'param1':'L', 'param2': 2.0, 'param3': 0.5},
    #             {'param1':'L', 'param2': 2.0, 'param3': 1.0},
    #             {'param1':'L', 'param2': 2.0, 'param3': 2.0}]