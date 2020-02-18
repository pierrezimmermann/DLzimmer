import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    InputLayer, Conv2D, Flatten, Dense
)


class CNN(Model):
    def __init__(self, name='cnn', **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((641, 432, 1),
                                      name='{}_input'.format(name))
        self.conv_layer_1 = Conv2D(200, (10, 10), activation=tf.nn.relu,
                                   name='{}_conv_1'.format(name))
        self.flatten = Flatten(name='{}_flatten'.format(name))
        self.dense_layer = Dense(2000, activation=tf.nn.relu,
                                 name='{}_dense'.format(name))
        self.output_layer = Dense(50, activation=tf.nn.softmax)

    def call(self, inputs):
        net = self.input_layer(inputs)
        net = self.conv_layer_1(net)
        net = self.flatten(net)
        net = self.dense_layer(net)
        net = self.output_layer(net)
        return net
