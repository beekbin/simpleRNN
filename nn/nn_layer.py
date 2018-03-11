from __future__ import print_function
from __future__ import division
import numpy as np
import math
import logging

class Layer(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        # activation function
        self.func = None
        # the output of current layer, usually is a vector of the activated result
        self.output = None
        self.input_layer = None
        self.next_layer = None
        self.lambda2 = 0
        return

    def init(self):
        """init the weight matrix"""
        pass

    def set_lambda2(self, l2):
        self.lambda2 = l2
        return

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        return

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        return

    def get_output(self):
        return self.output

    def get_size(self):
        return self.size

    def __str__(self):
        return "%d\t%s" % (self.size, self.name)

    def detail_info(self):
        fan_in = 0
        if self.input_layer is not None:
            fan_in = self.input_layer.get_size()

        if self.func is None:
            funcName = "None"
        else:
            funcName = self.func.get_name()
        msg = "[%d, %d],l2=%.5f, activation=[%s], %s" % (fan_in, self.size, self.lambda2, funcName, self.name)
        return msg


class InputLayer(Layer):
    def __init__(self, name, size):
        super(InputLayer, self).__init__(name, size)
        return

    def init(self):
        """do nothing"""
        return

    def feed(self, x):
        """for RNN, x is a list of integers (word index)"""
        self.output = x
        return

