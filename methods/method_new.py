from methods.MethodTemplate import MethodTemplate


class NewMethod(MethodTemplate):

    def method(self,signals,params = None):
        return signals
        


def instantiate_method():
    return NewMethod('denoising','a_method')
