from __future__ import print_function
from __future__ import division

import numpy as np
import math
import logging

def clip_gradient(x, x_min, x_max, threshold):
    """clip the gradient in 2 steps:
    (1) clip the outlier gradients;
    (2) normalize the gradient if its norm is too big;
    """
    np.clip(x, x_min, x_max)
    n = np.linalg.norm(x)
    if n > threshold:
        logging.info("begin to do norm clipping")
        x = n * (threshold/n)
    return


def myrandom_vector(d1):
    return np.random.randn(d1)

# init the weight with care
#  https://stats.stackexchange.com/questions/204114/deep-neural-network-weight-initialization?rq=1
#  https://arxiv.org/abs/1206.5533  Practical Recommendations for Gradient-Based Training of Deep Architectures
def sigmoid_init_weights(d1, d2):
    num = d1 * d2
    r = math.sqrt(6.0/num)
    tmp = np.random.uniform(-r, r, num)
    return tmp.reshape((d1, d2))


def tanh_init_weights(d1, d2):
    tmp = sigmoid_init_weights(d1, d2)
    tmp *= 4.0
    return tmp


def calc_softmax(z):
    tmp = np.exp(z)
    total = sum(tmp)
    return tmp/total


class LRScheduler:
    """a simple learning rate scheduler:
    lrt = (lr * tao)/max(t^a, tao)
    where lrt is the new learning rate;
          lr is the initial learning rate;
          tao is the initial steps: for the first $tao$ steps, the learning rate will be lr;
          a is usually around 1.0;
    """
    def __init__(self, lr, tao):
        self.lr = lr
        self.tao = tao
        self.a = 1.0
        return

    def set_alpha(self, alpha):
        self.a = alpha
        return    

    def get_lr(self, t):
        if self.a != 1.0:
            t = math.pow(t, self.a)

        if t <= self.tao:
            return self.lr
        
        return (self.lr * self.tao) / t    

    def __str__(self):
        info = "l0=%f, tao=%f, alpha=%f" % (self.lr, self.tao, self.a)
        return info 