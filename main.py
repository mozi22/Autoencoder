import tensorflow as tf
import input_pipeline as inpp
import losses_helper
import network

class Autoencoder:

    def __init__(self):
        alternate_global_step = tf.placeholder(tf.int32)

        self.MAX_ITERATIONS = 10000
        self.learning_rate = tf.train.polynomial_decay(0.0001, alternate_global_step,
                                                  self.MAX_ITERATIONS, 0.000001,
                                                  power=3)

        self.dataset = inpp.parse()
        self.iterator = self.dataset.make_initializable_iterator()

    def create_network(self):
        rainyImage = self.iterator.get_next()
        resulting_img, latent_space = network.create_network(rainyImage)

        l1_loss = losses_helper.reconstruction_loss_l1(resulting_img, rainyImage)
        loss_kl_shared = losses_helper.KL_divergence_loss(latent_space)

        loss = tf.reduce_mean(l1_loss + loss_kl_shared)

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.999).minimize(loss)

    def train(self):
        pass



obj = Autoencoder()
obj.create_network()