# simpleRNN
This simple nerual networks has an embedding layer, RNN layer, FC(fully connected) layer, and Softmax output layer.
 The RNN layer and FC layer can be stacked up to construct deeper neural networks.

With these layers, a __Seq2Seq__ model is built to learn and predict sequences of characters.

These layers and training process are implemented from scratch with __Python__ and __Numpy__ only.
I also wrote [an essay](https://github.com/beekbin/rnnEssay) about how to implement RNN.



# Construct a Seq2Seq nerual network

Following is an example code about building a seq2seq model, with one embedding layer, two RNN layers, 
and a Softmax output layer.


```python
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

    #3. add a RNN layer
    rnn = RNNLayer("rnn1", RNN_HIDDEN_DIM, MAX_BPTT_STEPS)
    rnn.set_lambda2(l2)
    nn.add_hidden_layer(rnn)

    #4. add another RNN layer
    rnn2 = RNNLayer("rnn2", RNN_HIDDEN_DIM2, MAX_BPTT_STEPS)
    rnn2.set_lambda2(l2)
    nn.add_hidden_layer(rnn2)

    #5. complete the nerual network
    nn.connect_layers()
    logging.info("NN information:\n" + nn.get_detail())
    return nn
```



# Run the toy
The toy example is digital sequences containing only the 10 digital numbers.
The sequences have two types: 

   (1) a sequence of odd digital numbers;
   
   (2) a sequence of even digital numbers;
   
### prerequisites
    * Python 2.7+
    * Numpy
    
### train the model
```bash
cd simpleRNN
python toy.py
```

Because of the convolution layer, the training process is very slow: takes around 4 hours to finish one echo. But the result is promising: after the training of the second epoch, it can get 98.54% correctness on testing set.
```console
[10-03-2018:23:33:26.729] INFO [toy.py:50] NN information:
[0, -1],l2=0.00000, activation=[None], word sequence input
[-1, 32],l2=0.00020, activation=[None], embedding
[32, 64],l2=0.00020, activation=[None], rnn1
[64, 10],l2=0.00000, activation=[None], word predict

[10-03-2018:23:33:26.741] INFO [toy.py:75] [develop] accuracy=0.1268, avg_cost=2.2649
[10-03-2018:23:33:28.302] INFO [simple_nn.py:93] cost = 1.872
[10-03-2018:23:33:29.911] INFO [simple_nn.py:93] cost = 0.651
[10-03-2018:23:33:31.520] INFO [simple_nn.py:93] cost = 0.121
[10-03-2018:23:33:33.132] INFO [simple_nn.py:93] cost = 0.053
[10-03-2018:23:33:34.740] INFO [simple_nn.py:93] cost = 0.033
[10-03-2018:23:33:36.348] INFO [simple_nn.py:93] cost = 0.024
...
[10-03-2018:23:34:45.625] INFO [toy.py:75] [develop] accuracy=0.9859, avg_cost=0.1087
[10-03-2018:23:34:47.28] INFO [simple_nn.py:93] cost = 0.003
```
