from __future__ import print_function
from __future__ import division

import numpy as np
import logging
import util
from nn_layer import Layer

class RNNLayer(Layer):
    def __init__(self, name, size, max_bptt_step):
        """A RNN Layer, size is hidden state."""
        super(RNNLayer, self).__init__(name, size)
        self.max_bptt_step = max_bptt_step
        self.size_in = -1 
        self.size_out = -1 

        # the activation function, it is tanh(x);
        #self.func = None

        # the W, U, V parameter matrixs
        self.weight_w = None
        self.weight_u = None
        self.weight_v = None
        self.bias = None

        self.delta_w = None
        self.delta_u = None
        self.delta_v = None
        self.delta_bias = None

        #hidden state, and its derivate
        self.h = None
        #input x for this RNN
        self.x = None
        self.delta_x = None
        #output y computed by RNN
        self.y = None

        return

    def init(self):
        """the training process will try to learn these four parameters"""
        self.size_in = self.input_layer.get_size()
        self.size_out = self.next_layer.get_size()

        self.weight_w = util.sigmoid_init_weights(self.size, self.size) 
        self.weight_u = util.sigmoid_init_weights(self.size_in, self.size)
        self.weight_v = util.sigmoid_init_weights(self.size, self.size_out)
        self.bias = util.myrandom_vector(self.size)

        self.delta_w = np.zeros(self.weight_w.shape)
        self.delta_u = np.zeros(self.weight_u.shape)
        self.delta_v = np.zeros(self.weight_v.shape)
        self.delta_bias = np.zeros(self.bias.shape)
        return

    def forward(self):
        """
        The forward pass, will genererate x, h, y for this input sequence X
        (1) assign the output of previous layer to self.x;
        (2) set each hidden state for each x[i];
        (3) set output for each x[i] based on h[i];
        """
        self.x = self.input_layer.get_output() 
        x = self.x

        T = x.shape[0] # T is total steps of input/output
        self.y = np.zeros((T, self.size_out))
        self.h = np.zeros((T+1, self.size))
        self.dh = np.zeros((T+1, self.size))

        # self.h[T] is treated as self.h[-1],
        #    will be used in the backward pass too, make sure its zero
        self.h[T] = np.zeros(self.size)

        for i in range(T):
            # z = X*U + h*w + b
            z = np.dot(x[i], self.weight_u)
            z += np.dot(self.h[i-1], self.weight_w)
            z += self.bias

            self.h[i] = np.tanh(z)
            self.y[i] = np.dot(self.h[i], self.weight_v)
        return

    def get_output(self):
        return self.y    

    def calc_input_delta(self):
        return self.delta_x    

    def backward(self):
        delta_y = self.next_layer.calc_input_delta()
        T = delta_y.shape[0]
        logging.debug("backward steps %s", T)
        # make sure T == self.y.shape[0]

        #0. hold the error to backpropagate to previous layer 
        self.delta_x = np.zeros(self.x.shape)

        #1. compute the derivatives of hidden state
        dh = np.zeros((T, self.size))
        for i in range(T):
            dh[i] = 1 - (self.h[i] ** 2)

        #2. for each output, run the BPTT process
        for i in range(T):
            self.delta_v += np.outer(self.h[i], delta_y[i])
            # delta = (dE/dy)(dy/dh)(dh/dz) = delta_y[i] * V * tanh'(z)
            delta = np.dot(delta_y[i], self.weight_v.T) * dh[i]

            begin = max(0, i - self.max_bptt_step)
            steps = range(begin, i+1)
            steps.reverse()
            for j in steps:
                self.delta_bias += delta
                self.delta_w += np.outer(self.h[j-1], delta)
                self.delta_u += np.outer(self.x[j], delta)

                # calculate delta_x to backpropagate error to the input layer
                self.delta_x[j] += np.dot(self.weight_u, delta) 

                # delta = delta * W * tanh'(z)
                delta = np.dot(self.weight_w, delta) * dh[j]
        
        #3. average up the derivatives
        self.delta_bias = self.delta_bias / T
        self.delta_w = self.delta_w / T
        self.delta_u = self.delta_u / T
        self.delta_v = self.delta_v / T
        #self.delta_x = self.delta_x / T

        return

    def clear_delta(self):
        self.delta_w.fill(0.0)
        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_bias.fill(0.0)
        return

    def clip_delta(self):
        x_min = 2
        x_max = 2
        norm_max = 5
        util.clip_gradient(self.delta_w, x_min, x_max, norm_max)
        util.clip_gradient(self.delta_u, x_min, x_max, norm_max)
        util.clip_gradient(self.delta_v, x_min, x_max, norm_max)
        util.clip_gradient(self.delta_bias, x_min, x_max, norm_max)
        return

    def update_weights(self, lr):
        if self.lambda2 > 0:
            self.delta_w += (self.lambda2 * self.weight_w)
            self.delta_u += (self.lambda2 * self.weight_u)
            self.delta_v += (self.lambda2 * self.weight_v)

        self.clip_delta()
        self.weight_w -= (lr * self.delta_w)
        self.weight_u -= (lr * self.delta_u)
        self.weight_v -= (lr * self.delta_v)
        self.bias -= (lr * self.delta_bias)

        self.clear_delta()
        return
