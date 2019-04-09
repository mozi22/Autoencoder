import tensorflow as tf


def encoder(input_image,scope='encoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        input_image = tf.pad(input_image, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv1 = tf.layers.conv2d(inputs=input_image, filters=16, kernel_size=4, strides=2, activation=tf.nn.leaky_relu, name='conv1')

        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='conv3')
        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='conv4')

        flattened = tf.layers.flatten(conv4)
        latent_space = tf.layers.dense(flattened, 4000, name='latent_space')

        return latent_space


def decoder(latent_space,scope='decoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):


        reshaped_flattened = tf.layers.dense(latent_space, 8192, name='reshaped_flattened')
        reshaped = tf.reshape(reshaped_flattened, [4, 8, 8, 128])

        convt_1 = tf.layers.conv2d_transpose(reshaped, filters=128, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_1')
        convt_2 = tf.layers.conv2d_transpose(convt_1, filters=64, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_2')
        convt_3 = tf.layers.conv2d_transpose(convt_2, filters=32, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_3')
        convt_4 = tf.layers.conv2d_transpose(convt_3, filters=16, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, name='convt_4')
        convt_5 = tf.layers.conv2d_transpose(convt_4, filters=3, kernel_size=1, strides=1, activation=tf.nn.tanh, name='convt_5')

        return convt_5

def create_network(input):
    latent_space = encoder(input)
    dec = decoder(latent_space)

    return dec, latent_space