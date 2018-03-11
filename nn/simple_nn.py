from __future__ import division
from __future__ import print_function
import numpy as np
import logging


class NNetwork(object):
    """A simple nerual network manager: 
       (1) manage the layers;
       (2) train/evaluate the network;
    """
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.error = 0.0
        self.error_count = 0
        self.log_interval = 1000
        return

    def set_log_interval(self, steps):
        self.log_interval = steps
        return

    def set_input(self, input_layer):
        self.input_layer = input_layer
        return

    def add_hidden_layer(self, hid_layer):
        self.hidden_layers.append(hid_layer)
        return

    def set_output(self, output_layer):
        self.output_layer = output_layer
        return

    def check(self):
        if self.input_layer is None:
            logging.error("input layer is None.")
            return False
        if self.output_layer is None:
            logging.error("output layer is None.")
            return False

        if len(self.hidden_layers) < 1:
            logging.error("hidden layers is empty.")
            return False
        return True

    def connect_layers(self):
        """set the input and output for the layers"""
        if not self.check():
            msg = "Failed to check neural network."
            print(msg)
            logging.error(msg)
            return

        # 1. set input layer
        pre_layer = self.input_layer
        for layer in self.hidden_layers:
            layer.set_input_layer(pre_layer)
            pre_layer = layer
        self.output_layer.set_input_layer(pre_layer)

        # 2. set output layer
        next_layer = self.output_layer
        for layer in reversed(self.hidden_layers):
            layer.set_next_layer(next_layer)
            next_layer = layer
        self.input_layer.set_next_layer(next_layer)

        # 3. call layer init
        self.input_layer.init()
        for layer in self.hidden_layers:
            layer.init()
        self.output_layer.init()

        return

    def forward(self, x):
        self.input_layer.feed(x)

        for layer in self.hidden_layers:
            layer.forward()

        self.output_layer.forward()
        return

    def log_cost(self, cost, num):
        self.error += cost
        self.error_count += num
        if self.error_count > self.log_interval:
            logging.info("cost = %.3f", self.error/self.error_count)
            self.error = 0.0
            self.error_count = 0

        return

    def backward(self, labels):
        self.output_layer.backward(labels)
        for layer in reversed(self.hidden_layers):
            layer.backward()

        cost = self.output_layer.calc_cost(labels)
        self.log_cost(cost, len(labels))
        return

    def update(self, lr):
        for layer in self.hidden_layers:
            layer.update_weights(lr)
        self.output_layer.update_weights(lr)
        return

    def train(self, x, y, lr):
        """train the model with single instance"""
        self.forward(x)
        self.backward(y)
        self.update(lr)
        return

    def __str__(self):
        msg = str(self.input_layer) + "\n"

        for layer in self.hidden_layers:
            msg += "%s\n" % (str(layer))

        msg += "%s\n" % (str(self.output_layer))
        return msg

    def get_detail(self):
        msg = "%s\n" % (self.input_layer.detail_info())

        for layer in self.hidden_layers:
            msg += ("%s\n" % (layer.detail_info()))

        msg += ("%s\n" % (self.output_layer.detail_info()))
        return msg

    def evaluate(self, x, y):
        self.forward(x)
        cost = self.output_layer.calc_cost(y)
        yy = self.output_layer.get_output()

        ii = np.argmax(yy, axis=1)
        correct = np.sum(y == ii)

        return correct, cost
