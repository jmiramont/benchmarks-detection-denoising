from methods.basic_methods import method_HT
import numpy as np
import pytest

@pytest.fixture
def dummy_input():
    dummy_input = np.random.rand(10,10)
    return dummy_input


def test_output_shape(dummy_input):
    output = method_HT(dummy_input, params=None)
    assert (output.shape == dummy_input.shape)
        


def test_output_type(dummy_input):
    output = method_HT(dummy_input, params=None)
    assert (type(output) is np.ndarray)



