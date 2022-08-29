from abc import ABC, abstractmethod
# import matlab.engine
from typeguard import is_typeddict
import numpy as np
# import sys
# sys.path.append("src\methods")

class MethodTemplate(ABC):

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def method(self):
        ...

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
    
    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, task):
        self._task = task

    def get_parameters(self):
        return (((),{}),) 

# class MatlabInterface():
#     def __init__(self,matlab_function_name):
#         self.matlab_function_name = matlab_function_name
#         self.eng = matlab.engine.start_matlab()
#         self.eng.eval("addpath('src/methods')")

#     def matlab_function(self,signal,*params):
#         all_params = list((signal,*params))
#         params = self.pre_parameters(*all_params)
#         fun_handler = getattr(self.eng, self.matlab_function_name)
#         return np.array(fun_handler(*params)[0].toarray())
        
#     def pre_parameters(self, *params):
#         params_matlab = list()
#         for param in params:
#             if isinstance(param,np.ndarray):
#                 params_matlab.append(matlab.double(vector=param.tolist()))
#             if isinstance(param,list) or isinstance(param,tuple):
#                 params_matlab.append(matlab.double(vector=list(param)))
#             if isinstance(param,float):
#                 params_matlab.append(matlab.double(param))
#             if isinstance(param,int):
#                 params_matlab.append(matlab.double(float(param)))
                
#         return params_matlab    

