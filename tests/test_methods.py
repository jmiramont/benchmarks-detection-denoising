import numpy as np
import subprocess
import importlib
from methods import *
import pytest
from benchmark_demo.SignalBank import SignalBank
from benchmark_demo.utilstf import add_snr

modules = dir()
modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
# print(modules)
global list_of_methods
list_of_methods = list()    
for mod_name in modules:
    mod = importlib.import_module('methods.' + mod_name)
    # print(mod)
    method = getattr(mod, 'return_method_instance')
    list_of_methods.append(method())

@pytest.fixture
def dummy_input():
    N = 256
    dummy_input = np.zeros((10,N))
    signal_bank = SignalBank(N)
    for i in range(10):
        dummy_input[i,:] = add_snr(signal_bank.linear_chirp(),15)
    return dummy_input


def test_methods_outputs(dummy_input):
        for method_instance in list_of_methods:
            method_id = method_instance.get_method_id()
            output = method_instance.method(dummy_input, params=None)
            assert (output.shape == dummy_input.shape), method_id +' output should be the same shape as input.'

# a_dummy_input = dummy_input()
# test_methods_outputs(a_dummy_input)
# test_methods_outputs(, )

# La idea acá es RECHAZAR un método que no cumpla con lo que tiene que cumplir para que no haya
# problemas corriendo el benchmark

# ¿Cómo testear CADA método? Ahora puedo listar los métodos, y correrlos. El tema es poder aplicar un
# test para cada uno. Se supone que pytest corre una función "test_*" y debería ser una función con esa
# firma.

# Después de esto hay que ver, por un lado, que las acciones de GitHub corran los tests y por otro
# que CORRAN EL BENCHMARK.



# def test_output_shape(dummy_input):


# def test_output_type(dummy_input):
#     output = method_HT(dummy_input, params=None)
#     assert (type(output) is np.ndarray)


