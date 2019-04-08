import tensorflow as tf

def reconstruction_loss_l1(prediction,gt):

	with tf.variable_scope('reconstruction_loss_l2'):
		loss = tf.reduce_mean(tf.square(prediction - gt))
		# epsilon = 1e-10
		# loss  = -tf.reduce_sum(gt * tf.log(epsilon + prediction) + (1 - gt) * tf.log(epsilon + 1 - prediction), axis=1)
		# loss = tf.reduce_mean(loss)
	return loss

def KL_divergence_loss(z_mu, z_sigma):

	with tf.variable_scope('kl_loss'):
		# loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(tf.square(z_sigma)) - 1, 1)
	# loss = tf.reduce_mean(KL_divergence)
		# mu_2 = tf.square(z_mu)
		# loss = tf.reduce_mean(mu_2)
		loss = -0.5 * tf.reduce_sum(1 + z_sigma - tf.square(z_mu) - tf.exp(z_sigma), axis=1)
		loss = tf.reduce_mean(loss)
	return loss
