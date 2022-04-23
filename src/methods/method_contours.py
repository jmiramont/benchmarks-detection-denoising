from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.
from methods.contours_utils import *

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'contour_filtering'
        self.task = 'denoising'
        

    def method(self, signals, params):
        """_summary_

        Args:
            signals (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(signals.shape) == 1:
            signals = np.resize(signals,(1,len(signals)))

        signals_output = np.zeros(signals.shape)
        for i, signal in enumerate(signals):
            if params is None:
                signals_output[i] = contours_filtering(signal)
            else:
                signals_output[i] = contours_filtering(signal, **params)

        return signals_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #   return [{'q': 0.95}, ]    
        # return [{'q': i} for i in (0.99, 0.95, 0.9)]
        