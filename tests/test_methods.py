import numpy as np
import importlib
import inspect
from methods import *
import pytest
from benchmark_demo.SignalBank import SignalBank
from benchmark_demo.utilstf import add_snr
from benchmark_demo.benchmark_utils import MethodTemplate

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
    signal_bank = SignalBank(N)
    signal = signal_bank.signal_linear_chirp()
    noise = np.random.randn(N,)
    signal += noise    
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
        assert isinstance(parameters[0][0], list) or isinstance(parameters[0][0], tuple), \
                    method_id +"'s positional arguments should be a list or a tuple."
        assert isinstance(parameters[0][1], dict), \
                    method_id +"'s keyword arguments should be a dictionary."



# Test the shape of the outputs for denoising methods
def test_denoising_methods_outputs_shape(dummy_input):
    for method_instance in list_of_methods:
        if method_instance.task == 'denoising':
            method_id = method_instance.id
            print(method_id)
            parameters = method_instance.get_parameters()
            for args, kwargs in parameters:
                output = method_instance.method(dummy_input, *args, **kwargs)
                assert (output.shape == dummy_input.shape), (method_id 
                                        +' output should have the same shape as input.')

#! This has to be corrected.
# Test the shape of the outputs for detection methods
# def test_detection_methods_outputs_shape(dummy_input):
#     for method_instance in list_of_methods:
#         if method_instance.task == 'detection':
#             method_id = method_instance.id
#             print(method_id)
#             parameters = method_instance.get_parameters()
#             for params in parameters:
#                 output = method_instance.method(dummy_input, params=params)
#                 assert (output.size == dummy_input.shape[0]), (method_id 
#                                         +' output should have one value per signal.')
                

# Test the type of the outputs
def test_methods_outputs_type(dummy_input):
        for method_instance in list_of_methods:
            if method_instance.task == 'denoising':
                method_id = method_instance.id
                print(method_id)
                # output = method_instance.method(dummy_input, params=None)
                parameters = method_instance.get_parameters()
                for args, kwargs in parameters:
                    output = method_instance.method(dummy_input, *args, **kwargs)
                    assert (type(output) is np.ndarray), (method_id 
                                                    +' output should be numpy.ndarray.')
            
            if method_instance.task == 'detection': # ! This has to be implemented.
                assert True

# test_denoising_methods_outputs_shape(dummy_input())