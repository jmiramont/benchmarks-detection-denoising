from ctypes import sizeof
import numpy as np
import importlib
import inspect
from methods import *
import pytest
from benchmark_tools.SignalBank import SignalBank
from benchmark_tools.utilstf import add_snr
from benchmark_tools.benchmark_utils import MethodTemplate
from benchmark_tools.Benchmark import get_args_and_kwargs

#---------------------------------------------------------------------------------------
""" This collects all the methods in the folder "methods" and make a list of them.
"""

modules = dir()
modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
global list_of_methods # Use with caution.
list_of_methods = list()    
for mod_name in modules:
    mod = importlib.import_module('methods.' + mod_name)
    classes_in_mod = inspect.getmembers(mod, inspect.isclass)
    for a_class in classes_in_mod:
        method_class = getattr(mod, a_class[0])
        class_parent = method_class.__bases__[0]
        if class_parent == MethodTemplate:
            elmetodo = method_class()
            list_of_methods.append(method_class())
#---------------------------------------------------------------------------------------

print([m.id for m in list_of_methods])

""" Here start the tests"""

# Generate dummy inputs for testing the methods.
@pytest.fixture
def dummy_input():
    N = 256
    Nsignals = 1
    dummy_input = np.zeros((Nsignals,N))
    signal_bank = SignalBank(N, return_signal=True)
    signal = signal_bank.signal_linear_chirp()
    noise = np.random.randn(N,)
    signal = signal + noise    
    return signal


# Test method inheritance of class MethodTemplate
def test_methods_inheritance():
    for method_instance in list_of_methods:
        method_id = method_instance.id
        assert isinstance(method_instance, MethodTemplate), (method_id 
                                                    +' should inherit MethodTemplate.')


# Test methods' ids and tasks.
def test_methods_attributes():
    for method_instance in list_of_methods:
        method_id = method_instance.id
        method_task = method_instance.task
        assert method_task in ['denoising','detection'], (method_id 
                                +"'s task should be either 'detection' or 'denoising'.")


# Test the shape of parameters
def test_methods_parameters():
    for method_instance in list_of_methods:
        method_id = method_instance.id
        parameters = method_instance.get_parameters()
        for params in parameters:
            args, kwargs = get_args_and_kwargs(params)
            assert isinstance(args, list) or isinstance(parameters[0][0], tuple), \
                    method_id +"'s positional arguments should be a list or a tuple."
            assert isinstance(kwargs, dict), \
                    method_id +"'s keyword arguments should be a dictionary."



# Test the shape of the outputs for denoising methods
def test_denoising_methods_outputs_shape(dummy_input):
    for method_instance in list_of_methods:
        if method_instance.task == 'denoising':
            method_id = method_instance.id
            print(method_id)
            parameters = method_instance.get_parameters()
            for params in parameters:
                args, kwargs = get_args_and_kwargs(params)
                output = method_instance.method(dummy_input, *args, **kwargs)
                assert (output.shape == dummy_input.shape), (method_id 
                                        +' output should have the same shape as input.')

# Test the shape of the outputs for detection methods
# def test_detection_methods_outputs_shape(dummy_input):
#     for method_instance in list_of_methods:
#         if method_instance.task == 'detection':
#             method_id = method_instance.id
#             print(method_id)
#             parameters = method_instance.get_parameters()
#             for params in parameters:
#                 args, kwargs = get_args_and_kwargs(params)
#                 output = method_instance.method(dummy_input, *args, **kwargs)
#                 assert sizeof(output) == 1 , 'Detection methods should output boolean values.'
                

# Test the type of the outputs
def test_denoising_methods_outputs_type(dummy_input):
        for method_instance in list_of_methods:
            method_id = method_instance.id
            print(method_id)
            parameters = method_instance.get_parameters()
            for params in parameters:
                args, kwargs = get_args_and_kwargs(params)
                output = method_instance.method(dummy_input, *args, **kwargs)
                if method_instance.task == 'denoising':
                    assert (type(output) is np.ndarray), (method_id 
                                                    +' output should be numpy.ndarray.')
            
                if method_instance.task == 'detection': # ! This has to be implemented.
                    assert type(output) is bool or type(output) is np.bool_ , 'Detection methods should output boolean values.'

# test_denoising_methods_outputs_shape(dummy_input())