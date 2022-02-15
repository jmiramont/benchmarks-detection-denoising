from methods.MethodTemplate import MethodTemplate


class NewMethod(MethodTemplate):

    def method(self,signals,params = None):
        return signals
        


def return_method_instance():
    return NewMethod('denoising','a_method')
