# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)

def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)


def maxpool2d(x, k=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, **unused_params):
    """Define variables of the model."""
    raise NotImplementedError()

  def run_model(self, unused_model_input, **unused_params):
    """Run model with given input."""
    raise NotImplementedError()

  def get_variables(self):
    """Return all variables used by the model for training."""
    raise NotImplementedError()

class SampleGenerator(BaseModel):
  def __init__(self):
    self.noise_input_size = 100

  def create_model(self, output_size, **unused_params):
    h1_size = 128
    self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
    self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')

    self.G_W2 = tf.Variable(xavier_init([h1_size, output_size]), name='g/w2')
    self.G_b2 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b2')

  def run_model(self, model_input, is_training=True, **unused_params):
    net = tf.nn.relu(tf.matmul(model_input, self.G_W1) + self.G_b1)
    output = tf.nn.sigmoid(tf.matmul(net, self.G_W2) + self.G_b2)
    return {"output": output}

  def get_variables(self):
    return [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

class SampleDiscriminator(BaseModel):
  def create_model(self, input_size, **unused_params):
    h1_size = 128
    self.D_W1 = tf.Variable(xavier_init([input_size, h1_size]), name='d/w1')
    self.D_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='d/b1')

    self.D_W2 = tf.Variable(xavier_init([h1_size, 1]), name='d/w2')
    self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name='d/b2')

  def run_model(self, model_input, is_training=True, **unused_params):
    net = tf.nn.relu(tf.matmul(model_input, self.D_W1) + self.D_b1)
    logits = tf.matmul(net, self.D_W2) + self.D_b2
    predictions = tf.nn.sigmoid(logits)
    return {"logits": logits, "predictions": predictions}

  def get_variables(self):
    return [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

"""
Junsu Lee GAN Model
"""
class TwoHiddenLayerGenerator(BaseModel):
  def __init__(self):
    self.noise_input_size = 100
  
  def create_model(self, output_size, **unused_params):
    h1_size = 258
    h2_size = 128
    
    self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
    self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')

    self.G_W2 = tf.Variable(xavier_init([h1_size, h2_size]), name='g/w2')
    self.G_b2 = tf.Variable(tf.zeros(shape=[h2_size]), name='g/b2')

    self.G_W3 = tf.Variable(xavier_init([h2_size, output_size]), name='g/w3')
    self.G_b3 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b3')

  def run_model(self, model_input, is_training=True, **unused_params):
    net = tf.nn.relu(tf.matmul(model_input, self.G_W1) + self.G_b1)
    net = tf.nn.relu(tf.matmul(net, self.G_W2) + self.G_b2)
    output = tf.nn.sigmoid(tf.matmul(net, self.G_W3) + self.G_b3)
    return {"output": output}

  def get_variables(self):
    return [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

class TwoHiddenDiscriminator(BaseModel):
  def create_model(self, input_size, **unused_params):
    h1_size = 258
    h2_size = 128

    self.D_W1 = tf.Variable(xavier_init([input_size, h1_size]), name='d/w1')
    self.D_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='d/b1')

    self.D_W2 = tf.Variable(xavier_init([h1_size, h2_size]), name='d/w2')
    self.D_b2 = tf.Variable(tf.zeros(shape=[h2_size]), name='d/b2')

    self.D_W3 = tf.Variable(xavier_init([h2_size, 1]), name='d/w3')
    self.D_b3 = tf.Variable(tf.zeros(shape=[1]), name='d/b3')

  def run_model(self, model_input, is_training=True, **unused_params):

    net = tf.nn.relu(tf.matmul(model_input, self.D_W1) + self.D_b1)
    net = tf.nn.relu(tf.matmul(net, self.D_W2) + self.D_b2)

    logits = tf.matmul(net, self.D_W3) + self.D_b3
    predictions = tf.nn.sigmoid(logits)
    return {"logits": logits, "predictions": predictions}

  def get_variables(self):
    return [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]


class CnnDiscriminator(BaseModel):
  def create_model(self, input_size, **unused_params):
    h1_size = 258
    h2_size = 128

    num_classes = 2
    l2_penalty=1e-8

    # print("input_size : ", input_size
    self.D_W1 = tf.Variable(xavier_init([3, 3, 1, 64]), name='d/w1')
    self.D_b1 = tf.Variable(tf.zeros(shape=[64]), name='d/b1')

    self.D_W2 = tf.Variable(xavier_init([3, 3, 64, 128]), name='d/w2')
    self.D_b2 = tf.Variable(tf.zeros(shape=[128]), name='d/b2')

    self.D_W3 = tf.Variable(xavier_init([3, 3, 128, 256]), name='d/w3')
    self.D_b3 = tf.Variable(tf.zeros(shape=[256]), name='d/b3')

    self.D_W4 = tf.Variable(xavier_init([12544, 1]), name='d/w4')
    self.D_b4 = tf.Variable(tf.zeros(shape=[1]), name='d/b4')

    # self.D_W5 = tf.Variable(xavier_init([1024, 1]), name='d/w5')
    # self.D_b5 = tf.Variable(tf.zeros(shape=[1]), name='d/b5')

    # self.D_W3 = tf.Variable(xavier_init([h2_size, 1]), name='d/w3')
    # self.D_b3 = tf.Variable(tf.zeros(shape=[1]), name='d/b3')

  def run_model(self, model_input, is_training=True, **unused_params):

    x = tf.reshape(model_input, shape=[-1, 50, 50, 1])
    net = conv2d(x, self.D_W1, self.D_b1)
    net = maxpool2d(net, k=2)

    net = conv2d(net, self.D_W2, self.D_b2)
    net = maxpool2d(net, k=2)

    net = conv2d(net, self.D_W3, self.D_b3)
    net = maxpool2d(net, k=2)
    
    print(net)
    # print(net.get_shape())
    # print("as_list : ", self.D_W2.get_shape().as_list()[0])
    print("as list : ", self.D_W4.get_shape().as_list())
  
    net = tf.reshape(net, [-1, self.D_W4.get_shape().as_list()[0]])
    # net = tf.nn.relu(tf.matmul(net, self.D_W4) + self.D_b4)
    logits = tf.matmul(net, self.D_W4) + self.D_b4
    # logits = slim.fully_connected(
    #     net, num_classes-1, activation_fn=None,
    #     weights_regularizer=slim.l2_regularizer(l2_penalty))

    predictions = tf.nn.sigmoid(logits)
    # logits = tf.matmul(model_input, self.D_W1) + self.D_b1
    # predictions = tf.nn.sigmoid(logits)
    return {"logits": logits, "predictions": predictions}

  def get_variables(self):
    return [self.D_W1, self.D_W2, self.D_W3, self.D_W4, self.D_b1, self.D_b2, self.D_b3, self.D_b4]