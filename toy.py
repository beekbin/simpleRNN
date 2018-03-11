from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import sys
import os
import logging
import random

from nn.nn_layer import InputLayer
from nn.softmax_layer import SoftmaxOutputLayer
from nn.rnn_layer import RNNLayer
from nn.simple_nn import NNetwork
from nn.embedding_layer import EmbeddingLayer
from nn.util import LRScheduler

WORD_DIM = 32
VOC_SIZE = 10
RNN_HIDDEN_DIM = 64
RNN_HIDDEN_DIM2 = 32
MAX_BPTT_STEPS = 15

def construct_nn(l2=0.0):
    seq_input = InputLayer("word sequence input", -1)
    seq_output = SoftmaxOutputLayer("word predict", VOC_SIZE)

    # 1. set input and output layers
    nn = NNetwork()
    nn.set_input(seq_input)
    nn.set_output(seq_output)

    #2. set embedding layer
    emb = EmbeddingLayer("embedding", VOC_SIZE, WORD_DIM)
    emb.set_lambda2(l2)
    nn.add_hidden_layer(emb)

    #3. set RNN layer
    rnn = RNNLayer("rnn1", RNN_HIDDEN_DIM, MAX_BPTT_STEPS)
    rnn.set_lambda2(l2)
    nn.add_hidden_layer(rnn)

    #4. add another RNN layer
    #rnn2 = RNNLayer("rnn2", RNN_HIDDEN_DIM2, MAX_BPTT_STEPS)
    #rnn2.set_lambda2(l2)
    #nn.add_hidden_layer(rnn2)

    #5. complete the nerual network
    nn.connect_layers()
    logging.info("NN information:\n" + nn.get_detail())
    return nn

def sequence_to_xy(seq):
    x = seq[0:-1]
    y = seq[1:]
    return x, y

def evaluate_it(nn, test_data, prefix):
    total = 0
    total_correct = 0
    total_cost = 0

    for i in range(len(test_data)):
        x, y = sequence_to_xy(test_data[i])
        correct, cost = nn.evaluate(x, y)

        total_correct += correct
        total += len(y)
        total_cost += cost

    accuracy = float(total_correct) / total
    avg_cost = total_cost / total

    msg = "[%s] accuracy=%.4f, avg_cost=%.4f" % (prefix, accuracy, avg_cost)
    logging.info(msg)
    return

def train_it(nn, train_data, lr):
    alist = range(len(train_data))
    random.shuffle(alist)

    num = 0
    for i in alist:
        x, y = sequence_to_xy(train_data[i])
        nn.train(x, y, lr)
        num += 1
        if num % 1000 == 0:
            logging.info("num=%d sentences" % num)

    return

def gen_fake_data():
    #1. generate tranining data
    train_dat = []
    train_dat.append([1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7,1,3])
    train_dat.append([2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8])

    #2. generate evaluating dat
    eval_dat = []
    seq = [1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7]
    eval_dat.append(seq) 
    seq = [1,3,5,7]
    eval_dat.append(seq)
    seq = [2,4,6,8]
    eval_dat.append(seq)
    seq =[2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8] 
    eval_dat.append(seq)
    seq = [1,3,5,7,1,3,5,7,2,4,6,8,2,4,6,8,2,4,6,8]
    eval_dat.append(seq)

    return train_dat, eval_dat

def main():
    l2 = 0.0002
    nn = construct_nn(l2)
    nn.set_log_interval(2000)

    train_dat, eval_dat = gen_fake_data()
    logging.info("train_dat=%d, eval_dat=%d", len(train_dat), len(eval_dat))
    lr = 0.008
    tao = 1500.0
    lrs = LRScheduler(lr, tao)

    epochs = 9900
    for i in range(epochs):
        lr = lrs.get_lr(i)
        train_it(nn, train_dat, lr)
        if i % 1000 == 0:
            evaluate_it(nn, eval_dat, "develop")

    evaluate_it(nn, eval_dat, "develop")        
    return

def setup_log():
    logfile = "./log/train.%s.log" % (os.getpid())
    if len(sys.argv) > 1:
        logfile = sys.argv[1]
    print("logfile=%s"%(logfile))
    logging.basicConfig(#filename=logfile
            format='[%(asctime)s.%(msecs)d] %(levelname)-s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%d-%m-%Y:%H:%M:%S',
            level=logging.INFO)
    return


if __name__ == "__main__":
    setup_log()
    sys.exit(main())

