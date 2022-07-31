from methods.benchmark_utils import MethodTemplate
# You must import the MethodTemplate abstract class.
from methods.contours_utils import *

class NewMethod(MethodTemplate):

    def __init__(self):
        self.id = 'contour_filtering'
        self.task = 'denoising'
        

    def method(self, signal, params):
        """_summary_

        Args:
            signals (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        if params is None:
            signal_output = contours_filtering(signal)
        else:
            signal_output = contours_filtering(signal, **params)

        return signal_output

    # def get_parameters(self):            # Use it to parametrize your method.
    #   return [{'q': 0.95}, ]    
        # return [{'q': i} for i in (0.99, 0.95, 0.9)]
        