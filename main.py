import tensorflow as tf
import input_pipeline as inpp
import losses_helper
from datetime import datetime
import network
import os
import keras


class Autoencoder:

    def __init__(self):
        self.ckpt_folder = './ckpt_mnist/'

        self.global_step = tf.get_variable(
                            'global_step', [],
                            initializer=tf.constant_initializer(0), trainable=False)

        self.alternate_global_step = tf.placeholder(tf.int32)

        self.MAX_ITERATIONS = 200000
        self.learning_rate = tf.train.polynomial_decay(0.00001, self.alternate_global_step,
                                                  self.MAX_ITERATIONS, 0.000001,
                                                  power=3)

        train_images, train_labels, test_images, test_labels = self.get_mnist_dataset()
        images = tf.convert_to_tensor(train_images)
        images = tf.expand_dims(images, axis=-1)
        images = tf.to_float(images)

        self.dataset = tf.data.Dataset.from_tensor_slices(images).repeat().shuffle(buffer_size=50).batch(4,True)


        # self.dataset = inpp.parse()
        self.iterator = self.dataset.make_initializable_iterator()


    def get_mnist_dataset(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        return train_images, train_labels, test_images, test_labels

    def create_network(self):
        self.input_image = self.iterator.get_next()


        # self.input_image = tf.image.resize_images(self.input_image, [28, 28])

        self.resulting_img, self.latent_space, z_mean, z_std = network.create_network(self.input_image)
        self.l1_loss = losses_helper.reconstruction_loss_l1(self.resulting_img, self.input_image)
        self.loss_kl_shared = losses_helper.KL_divergence_loss(z_mean, z_std)
        self.loss = self.l1_loss + self.loss_kl_shared

        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.999).minimize(self.loss)
        self.create_tensorboard()

    def get_summary_ops(self):
        train_summaries = []
        train_summaries.append(tf.summary.image('InputImage', self.input_image))
        train_summaries.append(tf.summary.image('ResultImage', self.resulting_img))
        train_summaries.append(tf.summary.scalar('KL_div', tf.reduce_mean(self.loss_kl_shared)))
        train_summaries.append(tf.summary.scalar('L1_Loss', self.l1_loss))
        for var in tf.trainable_variables():
            train_summaries.append(tf.summary.histogram(var.op.name, var))

        return train_summaries

    def create_tensorboard(self):
        self.iteration = 0

        self.saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
            device_count={'GPU': 1}
        )
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        self.sess.run(self.iterator.initializer)
        self.summary_writer = tf.summary.FileWriter(self.ckpt_folder, self.sess.graph)
        self.summary_op = tf.summary.merge(self.get_summary_ops())

        self.loop_start = tf.train.global_step(self.sess, self.global_step)
        self.loop_stop = self.loop_start + self.MAX_ITERATIONS


    def train(self):

        for step in range(self.loop_start, self.loop_stop + 1):
            _, loss = self.sess.run([self.opt, self.loss], feed_dict={
                self.alternate_global_step: self.iteration
            })

            format_str = ('%s: step %d, g_loss = %.15f')
            print((format_str % (datetime.now(), step, loss)))

            if step % 100 == 0:
                summmary = self.sess.run(self.summary_op, feed_dict={
                    self.alternate_global_step: self.iteration
                })
                self.summary_writer.add_summary(summmary, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0:
                checkpoint_path = os.path.join(self.ckpt_folder, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=step)

            self.iteration += 1
        self.summary_writer.close()

obj = Autoencoder()
obj.create_network()
obj.train()