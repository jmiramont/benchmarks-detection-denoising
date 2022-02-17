import numpy as np
import importlib
from methods import *
import pytest
from benchmark_demo.SignalBank import SignalBank
from benchmark_demo.utilstf import add_snr
from methods.MethodTemplate import MethodTemplate

#-------------------------------------------------------------------------------------------------------
""" This collects all the methods in the folder/ module "methods" and make a global list of them."""
modules = dir()
modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
global list_of_methods # Use with caution.
list_of_methods = list()    
for mod_name in modules:
    mod = importlib.import_module('methods.' + mod_name)
    # print(mod)
    method = getattr(mod, 'instantiate_method')
    list_of_methods.append(method())


#-------------------------------------------------------------------------------------------------------
""" Here start the tests"""

# Generate dummy inputs for testing the methods.
@pytest.fixture
def dummy_input():
    N = 256
    dummy_input = np.zeros((5,N))
    signal_bank = SignalBank(N)
    for i in range(5):
        dummy_input[i,:] = add_snr(signal_bank.linear_chirp(),15)
    return dummy_input


# Test method inheritance of class MethodTemplate
def test_methods_inheritance():
        for method_instance in list_of_methods:
            method_id = method_instance.get_method_id()
            assert isinstance(method_instance, MethodTemplate), method_id +' should inherit MethodTemplate.'
            

# Test the shape of the outputs
def test_methods_outputs_shape(dummy_input):
        for method_instance in list_of_methods:
            if method_instance.get_task() == 'denoising':
                method_id = method_instance.get_method_id()
                print(method_id)
                output = method_instance.method(dummy_input, params=None)
                assert (output.shape == dummy_input.shape), method_id +' output should have the same shape as input.'

            if method_instance.get_task() == 'detection':
                method_id = method_instance.get_method_id()
                print(method_id)
                output = method_instance.method(dummy_input, params=None)
                assert (output.shape[0] == dummy_input.shape[0]), method_id +' output shape[0] should be the same as input.shape[0].'
                assert all(isinstance(i,bool) for i in output), method_id +' output should be an array of booleans.'




# Test the type of the outputs
def test_methods_outputs_type(dummy_input):
        for method_instance in list_of_methods:
            method_id = method_instance.get_method_id()
            print(method_id)
            output = method_instance.method(dummy_input, params=None)
            assert (type(output) is np.ndarray), method_id +' output should be numpy.ndarray.'