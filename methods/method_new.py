from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.

""" Put here all the functions that your method uses or import them.
Remark: Make sure that this file starts with "method_".
|
|
|
|
| def a_function_of_my_method(signal,params):
|   ...
|
|
|
"""


""" Create here a new class that will encapsulate your method.
This class should inherit the abstract class MethodTemplate.
By doing this, you must then implement the class method: 

method(self, signal, params)

which should receive the signals and any parameters that you desire.
"""
class NewMethod(MethodTemplate):

    def method(self,signals,params = None): # Implement this method.
        return signals[2::]
        

""" Here you can define the task your new method is devoted to 
(detecting or denoising). You can also choose a method name.
"""
method_task = 'denoising'
method_name = 'a_new_method'

def instantiate_method():
    return NewMethod(method_task,method_name)
