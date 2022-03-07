from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.
from methods.contours_utils import *


def contour_selection(signal,params):
    # if len(signal.shape) == 1:
    #     signal = np.resize(signal,(1,len(signal)))
    
    x_hat, _, _, _ = contours_filtering(signal,) 
    return x_hat

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'contour_filtering'
        self.task = 'denoising'
        

    def method(self,signals,params = None):
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            signals_output[i] = contour_selection(signal,params)

        return signals_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return [None,]
        