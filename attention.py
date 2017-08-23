import numpy as np
import tensorflow as tf

# Contact dhruvramani98@gmail.com for explanation and notes.

class Attention(object):
    def __init__(self, output_vectors):
        tf.set_random_seed(1234)
        self.H = output_vectors
        self.Wh, self.Walpha, self.Wp, self.Wx = self.init_weights()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def init_weights(self):
        d = self.H.get_shape().as_list()[0]
        Wh = self.weight_variable([d, d], name='Wh')
        Walpha = self.weight_variable([1, d], name='Walpha')
        Wp = self.weight_variable([1, d], name='Wp')
        Wx = self.weight_variable([1, d], name='Wx')
        return Wh, Walpha, Wp, Wx

    def applyAttention(self):
        M = tf.nn.tanh(tf.matmul(self.Wh, self.H))
        alpha = tf.nn.softmax(tf.matmul(self.Walpha, M)) 
        r = tf.matmul(self.H, tf.transpose(alpha))
        h_ = tf.nn.tanh(tf.matmul(self.Wp, r) + tf.matmul(Wx, self.H[:, -1]))
        return h_
