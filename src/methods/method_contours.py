from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.
from methods.contours_utils import *


def contour_selection(signal,params):
    # if len(signal.shape) == 1:
    #     signal = np.resize(signal,(1,len(signal)))
    
    x_hat, _, _, _ = contours_filtering(signal,) 
    return x_hat



""" Create here a new class that will encapsulate your method.
This class should inherit the abstract class MethodTemplate.
By doing this, you must then implement the class method: 

def method(self, signal, params)

which should receive the signals and any parameters
that you desire to pass to your method.You can use this file as an example.
"""

class NewMethod(MethodTemplate):

    def method(self,signals,params = None):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            signals_output[i] = contour_selection(signal,params)

        return signals_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return [None,]
        

""" Here you can define the task your new method is devoted to 
(detecting or denoising). You can also choose a method name.
"""
method_task = 'denoising' # 'denoising' or 'detection'
method_name = 'contour_filtering'

def instantiate_method():
    return NewMethod(method_task,method_name)
