import tensorflow as tf

def reconstruction_loss_l1(prediction,gt):

	with tf.variable_scope('reconstruction_loss_l1'):

		flattened_predictions = tf.layers.flatten(prediction)
		flattened_gt = tf.layers.flatten(gt)

		# loss = -tf.reduce_sum(flattened_gt * tf.log(1e-8 + flattened_predictions) + (1-flattened_gt) * tf.log(1e-8 + 1 - flattened_predictions),1)
		loss = tf.reduce_mean(flattened_gt - flattened_predictions)

	return loss

def KL_divergence_loss(z_mu, z_std):

	with tf.variable_scope('kl_loss'):


		loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_std) - tf.log(tf.square(z_std)) - 1, 1)
		# mu_2 = tf.square(z_mu)
		# loss = tf.reduce_mean(mu_2)

	return loss
