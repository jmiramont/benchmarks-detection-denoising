from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.

"""
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
"""
from numpy import pi as pi


""" Put here all the functions that your method uses. """
def a_function_of_my_method(signals, params):
    return signals * params[0] + params[1]


""" Create here a new class that will encapsulate your method.
This class should inherit the abstract class MethodTemplate.
By doing this, you must then implement the class method: 

def method(self, signal, params)

which should receive the signals and any parameters
that you desire to pass to your method.You can use this file as an example.
"""

class NewMethodWithParameters(MethodTemplate):
    def __init__(self):
        self.id = 'a_new_method_with_parameters'
        self.task = 'denoising'  # Should be either 'denoising' or 'detection'

# This simple method returns the same set of signals multiplied by a number
# and summing a constant. These values are passed as input parameters.

    def method(self, signals, params = None): # Implement this method.
        return a_function_of_my_method(signals, params)


# Let us define a lists of parameters to pass to our method.
# Here, we define a list of pairs of values to pass the method.
# This will make the method is tested two times:
# First, using params = [1, 2], and then using params [3, 4].

    def get_parameters(self):            # Use it to parametrize your method.
         return ((1, 2),(3, 4),)
