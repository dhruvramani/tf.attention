# Attention in Tensorflow 
Plug-and-Play version of Attention implemented in Tensorflow (1.2.1).

## Installation
Install [Tensorflow](https://www.tensorflow.org/install/) for your system.

## Usage
You can use this file with your code in the following manner :
```python

    from attention import Attention
    ...
    output_vectors = lstm()
    atta = Attention(output_vectors)
    att_vec = atta.applyAttention()
    output = tf.nn.softmax(tf.matmul(Wh, att_vec) + b)
    ...
```

## Current Status 
Haven't checked, will be running it on something soon, hopefully.