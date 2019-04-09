import tensorflow as tf


def encoder(input_image,scope='encoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=input_image, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='conv2')

        flattened = tf.layers.flatten(conv2)
        latent_space = tf.layers.dense(flattened, 100, name='latent_space')


        return latent_space


def decoder(latent_space,scope='decoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):

        reshaped_flattened = tf.layers.dense(latent_space, 1568, name='reshaped_flattened')
        reshaped = tf.reshape(reshaped_flattened, [4, 7, 7, 32])

        convt_1 = tf.layers.conv2d_transpose(reshaped, filters=32, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_1')
        convt_2 = tf.layers.conv2d_transpose(convt_1, filters=16, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_2')
        convt_3 = tf.layers.conv2d_transpose(convt_2, filters=1, kernel_size=1, strides=1, activation=tf.nn.tanh, name='convt_3')
        return convt_3

def create_network(input):
    latent_space = encoder(input)
    dec = decoder(latent_space)

    return dec, latent_space