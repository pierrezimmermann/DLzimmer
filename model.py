import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer, Conv2D, Flatten, Dense, MaxPool2D, Dropout
)
from tensorflow.keras.constraints import max_norm


class CNN(Sequential):
    def __init__(self, name='cnn', dropout_rate=0.5, **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((35, 35, 3),
                                      name='{}_input'.format(name))
        self.conv_layer_1 = Conv2D(10, (5, 5), activation=tf.nn.relu,
                                   name='{}_conv_1'.format(name))
        self.pool_layer_1 = MaxPool2D(name='{}_pool_1'.format(name))
        self.conv_layer_2 = Conv2D(20, (5, 5), activation=tf.nn.relu,
                                   name='{}_conv_2'.format(name))
        self.pool_layer_2 = MaxPool2D(name='{}_pool_2'.format(name))
        self.flatten = Flatten(name='{}_flatten'.format(name))
        self.dropout = Dropout(dropout_rate,
                               name='{}_dropout'.format(name))
        self.dense_layer_1 = Dense(128, activation=tf.nn.relu,
                                   name='{}_dense_1'.format(name))
        self.dense_layer_2 = Dense(500, activation=tf.nn.relu,
                                   name='{}_dense_2'.format(name))
        self.output_layer = Dense(50, activation=tf.nn.softmax)

    def call(self, inputs):
        net = self.input_layer(inputs)
        net = self.conv_layer_1(net)
        net = self.pool_layer_1(net)
        net = self.conv_layer_2(net)
        net = self.pool_layer_2(net)
        net = self.flatten(net)
        net = self.dropout(net)
        net = self.dense_layer_1(net)
        net = self.dense_layer_2(net)
        net = self.output_layer(net)
        return net
