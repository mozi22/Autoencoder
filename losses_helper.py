import tensorflow as tf

def reconstruction_loss_l1(prediction,gt):

	with tf.variable_scope('reconstruction_loss_l2'):
		loss = tf.reduce_mean(tf.square(prediction - gt))

	return loss

def KL_divergence_loss(z_mu, z_sigma):

	with tf.variable_scope('kl_loss'):
		loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(tf.square(z_sigma)) - 1, 1)
	# loss = tf.reduce_mean(KL_divergence)
		# mu_2 = tf.square(z_mu)
		# loss = tf.reduce_mean(mu_2)

	return loss
