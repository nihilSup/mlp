"""
Describes neyron and mlp net
"""
import math

class Neyron(object):
    """simple neyron with weights, bias and activation function.
    Neyron(act_func, num_inputs)
    """
    def __init__(self, act_func, num_inputs):
        self.act_func = act_func
        self.weights = [0.5] * num_inputs
        self.bias = 0.5
    def process_input(self, x_values):
        'calculation z of weights and inputs'
        return sum([weight * inp
                    for weight, inp in zip(self.weights, x_values)],
                   self.bias)
    def __call__(self, input_vector):
        return self.act_func(self.process_input(input_vector))
    def build_sigmoid(self, num_inputs):
        'builds sigmoid neyron'
        return Neyron(sigmoid, num_inputs)
def sigmoid(z_value):
    "1 / (1 + exp(-z))"
    return 1 / (1 + math.exp(-z_value))
class Net(object):
    """
    >>> Net(list of layers)
    """
    def __init__(self, layers):
        self.layers = layers
