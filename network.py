import tensorflow as tf
import keras

def conv(name,inputs, filters, kernel_size, stride, activation=tf.nn.leaky_relu, pad=0):

    inputs = tf.pad(inputs, [[0,0], [pad, pad], [pad, pad], [0,0]])

    layer = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            strides=stride,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                            activation=activation,
                            name=name)

    return layer

def conv_transpose(name,inputs, filters, kernel_size, stride, activation=tf.nn.leaky_relu):

    layer = tf.layers.conv2d_transpose(inputs=inputs,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding='SAME',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       strides=stride,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                       activation=activation,
                                       name=name)

    return layer

def dense_layer(name, inputs, units, activation=tf.nn.leaky_relu):
    layer = tf.layers.dense(inputs,
                            units=units,
                            activation=activation,
                            name=name)

    return layer

def encoder(input_image,scope='encoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        conv1 = conv(name='conv1', inputs=input_image, filters=32, kernel_size=7, stride=2, pad=3)
        conv2 = conv(name='conv2', inputs=conv1, filters=64, kernel_size=3, stride=2, pad=1)
        conv3 = conv(name='conv3', inputs=conv2, filters=128, kernel_size=3, stride=2, pad=1)

        flattened = tf.layers.flatten(conv3)

        z_mean = tf.layers.dense(flattened, 2048, name='z_mean')
        z_std = tf.layers.dense(flattened, 2048, name='z_std')



        return z_mean, z_std

def create_latent_space(z_mean, z_std, scope='create_latent_space', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):



        z_random = tf.random_normal(shape=tf.shape(z_mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        latent_space = z_mean + z_std * z_random

        return latent_space


def decoder(latent_space,scope='decoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):

        dense_t = tf.layers.dense(latent_space, 32768, name='dense_t')
        latent_space = tf.reshape(dense_t,[4, 16, 16, 128])


        conv_tran4 = conv_transpose(name='conv4_transpose', inputs=latent_space, filters=128, kernel_size=3, stride=2)
        conv_tran3 = conv_transpose(name='conv3_transpose', inputs=conv_tran4, filters=64, kernel_size=7, stride=2)
        conv_tran2 = conv_transpose(name='conv2_transpose', inputs=conv_tran3, filters=32, kernel_size=1, stride=2)
        conv_tran1 = conv_transpose(name='conv1_transpose', inputs=conv_tran2, filters=3, kernel_size=1, stride=1, activation=tf.nn.tanh)

        return conv_tran1

def create_network(input):
    z_mean, z_std = encoder(input)
    latent_space = create_latent_space(z_mean, z_std)
    dec = decoder(latent_space)

    return dec, latent_space, z_mean, z_std