import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        
        # generator
        g_hidden_size = [256],
        g_output_size = [28, 28],
        g_hidden_activation = tf.relu,
        g_output_activation = None,
        z_size = 128,

        # discriminator
        d_hidden_size = [256],
        d_output_size = 1,
        d_hidden_activation = tf.relu,
        d_output_activation = tf.sigmoid,

        # Training related parameters
        batch_size = 64,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5,
        
        # Beam Search
        beam_width = 10,
        length_penalty_weight = 2.1
        )
