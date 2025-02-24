import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight for attention
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
l
