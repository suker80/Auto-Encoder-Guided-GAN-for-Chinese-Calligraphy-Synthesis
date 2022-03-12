import tensorflow.contrib as contrib
import tensorflow as tf
import numpy as np

def batch_norm(input):
    return contrib.layers.batch_norm(inputs=input,decay=0.9,scale=True,epsilon=1e-5,updates_collections=None)

def conv2d(input,num_output,activation=None,kernel=[4,4],stride=[2,2],padding='SAME'):
    return tf.layers.conv2d(inputs=input,
                            filters=num_output,
                            kernel_size=kernel,
                            strides=stride,
                            padding=padding,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
def leaky_relu(x,leak=0.2):
    return tf.nn.leaky_relu(x)
def deconv2d(input,num_output,kernel=[4,4],stride=[2,2],padding='SAME'):
    return tf.layers.conv2d_transpose(inputs=input,
                                      filters=num_output,
                                      kernel_size=kernel,
                                      strides=stride,
                                      padding=padding,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

def relu(x):
    return tf.nn.relu(x)

def LBC(input,output_size,kernel=[4,4],stride=[2,2],padding='SAME'):
    return leaky_relu(batch_norm(conv2d(input,output_size,kernel=kernel,stride=stride,padding='SAME')))

def RBC(input,output_size,kernel=[4,4],stride=[2,2],padding='SAME'):
    return relu(batch_norm(conv2d(input,output_size,kernel=kernel,stride=stride,padding='SAME')))

def RBD(input,output_size):
    return relu(batch_norm(deconv2d(input,output_size,kernel=[4,4],stride=[2,2],padding='SAME')))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def dropout(x):
    return tf.nn.dropout(x,0.2)

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
