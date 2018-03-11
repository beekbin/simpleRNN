from __future__ import print_function
from __future__ import division
import numpy as np
from nn_layer import Layer
import logging


class EmbeddingLayer(Layer):
    def __init__(self, name, num, dim):
        """num: size of vocabulary;
           dim: dimension of the word vector;
        """
        super(EmbeddingLayer, self).__init__(name, dim)
        self.num = num
        self.dim = dim
        self.weights = None
        self.word_index = -1
        self.delta = None #self.delta will get from next layer.
        return

    def init(self):
        self.weights = np.zeros((self.num, self.dim))
        for i in range(self.num):
            self.weights[i] = np.random.uniform(-1, 1, self.dim)
        return

    def forward(self):
        """the input of this layer is the index of the word."""
        indata = self.input_layer.get_output()

        if np.max(indata) > self.num or np.min(indata) < 0:
            logging.error("EmbeddingLayer[%s] out of index [%s Vs. %s]." % (self.name, indata, self.num))
            return

        self.word_index = indata
        self.output = self.weights[indata]
        return

    def backward(self):
        self.delta = self.next_layer.calc_input_delta()
        return

    def update_weights(self, lr):
        """update each word vector.
        TODO: average the deltas for the same word;
        """
        for i in range(len(self.word_index)):
            idx = self.word_index[i]
            delta = self.delta[i]

            if self.lambda2 > 0:
                delta += (self.lambda2 * self.weights[idx])
            self.weights[idx] -= (lr * delta)    
        return
