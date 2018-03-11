from __future__ import print_function
from __future__ import division
import numpy as np
import math
import logging
from nn_layer import Layer


class SoftmaxOutputLayer(Layer):
    """softmax output
    the input is a 2-D matrix: N x M;
    where N is the number of instances,
          M is the number of classes;
    """
    def __init__(self, name, size):
        super(SoftmaxOutputLayer, self).__init__(name, size)
        self.delta = None
        self.output = None
        return

    def init(self):
        """do nothing"""
        #self.output = np.zeros(self.size)
        return

    def get_output(self):
        return  self.output

    def forward(self):
        x = self.input_layer.get_output()
        xx = x.T
        ex = np.exp(xx - np.max(xx))
        tmp = ex/ex.sum(axis=0) 
        self.output = tmp.T 
        logging.debug("x=%s, output=%s", x.shape, self.output.shape)
        return

    def backward(self, labels):
        #self.delta = self.output[:, labels] - 1
        self.delta = np.copy(self.output)
        for i in range(len(labels)):
            self.delta[i, labels[i]] -= 1

        return

    def calc_input_delta(self):
        return self.delta

    def calc_cost(self, labels):
        """calculate all the costs, and return the sum of the costs"""
        result = 0.0
        for i in range(len(labels)):
            y = self.output[i, labels[i]]
            result += calc_log(y)
        logging.debug("cost = %.4f", result/len(labels))
        return result

    def update_weights(self, lr):
        """do nothing"""
        return

def calc_log(y):
    if y < 0.00000001:
        return 10000000
    elif y > 0.99999999:
        return 0
    return -1 * math.log(y)