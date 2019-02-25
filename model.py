import sys
sys.path.append('submodule/')

import numpy as np
import tensorflow as tf

import generator_impl, discriminator_impl


def gan(features, labels, mode, params):
    dtype = params['dtype']
    
    g_hidden_size = params['hidden_size']
    g_output_size = params['output_size']
    g_hidden_activation = params['g_hidden_activation']
    g_output_activation = parmas['g_output_activation']
    z_size = params['z_size']

    d_hidden_size = params['d_hidden_size']
    d_output_size = params['d_output_size']
    d_hidden_activation = params['d_hidden_activation']
    d_output_activation = params['d_output_activation']

    batch_size = params['batch_size']


    # Sample z and generate synthetic data
    with tf.variable_scope('GeneratorScope'):
        z = tf.random.uniform([batch_size, z_size], dtype = dtype, name = 'z')
        generator = generator_impl.Generator(g_hidden_size, g_output_size, g_hidden_activation, g_output_activation)

        g_out = generator(z)

    # Sample real data discriminate
    with tf.variable_scope('DiscriminatorScope'):
        x = features['x']
        discriminator = discriminator_impl.Discriminator(d_hidden_size, d_output_size, d_hidden_activation, d_output_actication)
        d_out =  discriminator(x)

    
    # Predict mode

    # Train mode
    # Prediction of discriminator
    prediction_real = tf.round(d_out)
    prediction_syn = tf.round(discriminator(g_out))

    # Define loss function for generator and discriminator
    loss_g = -tf.reduce_mean(tf.log(g_out))
    loss_d = -tf.reduce_mean(tf.log(d_out) + tf.log(1 - discriminator(g_out)))

    eval_metric_ops = {
            'accuracy_real': tf.metrics.accuracy(tf.ones([batch_size, 1]), predictions = prediction_real),
            'accuracy_syn': tf.metrics.accuracy(tf.zeros([batch_size, 1]), prediction_syn)
            }


    tf.summary.scalar('loss_g', loss_g)
    tf.summary.scalar('loss_d', loss_d)
    
    learning_rate = params['learning_rate']
    learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), 500, params['decay_step'], staircase = True)
    optimizer_g = tf.train.AdamOptimizer(learning_rate)
    optimizer_d = tf.train.AdamOptimizer(learning_rate)

    trainable_variables_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'GeneratorScope')
    trainable_variables_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'DiscriminatorScope')

    grad_and_var_d = optimizer.compute_gradients(loss_d, tf.trainable_variables())
    grad_and_var_g = optimizer.comepute_gradients(loss)
    
    # Add histogram summary for gradient
    for grad, var in grad_and_var_d:
        tf.summary.histogram(var.name + '/gradient', grad)
    for grad, var in grad_and_var_g:
        tf.summary.histogram(var.name + '/gradient', grad)

    train_op = optimizer.apply_gradients(grad_and_var, global_step = tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
