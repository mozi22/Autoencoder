import tensorflow as tf

def reconstruction_loss_l1(prediction,gt):

	with tf.variable_scope('reconstruction_loss_l2'):
		loss = tf.reduce_mean(tf.square(prediction - gt))
		loss = tf.losses.compute_weighted_loss(loss, weights=1000)
		# loss = tf.Print(loss, [loss], 'recon')
	return loss

def KL_divergence_loss(z_mu, z_sigma):

	with tf.variable_scope('kl_loss'):
		loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(tf.square(z_sigma)) - 1, 1))
		loss = tf.losses.compute_weighted_loss(loss, weights=1)
		# loss = tf.Print(loss,[loss],'kl_div')
	return loss
