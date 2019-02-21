import sys
sys.path.append('submodule/')

import numpy as np
import tensorflow as tf


def model_name(features, labels, mode, params):
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']
    bucket_sizes = params['bucket_sizes']
    
    
    def embed_op(inputs, params):
        if params['embedding'] == None:
            embedding = tf.get_variable('embedding', [params['voca_size'], params['hidden_size']], dtype = params['dtype'])
        else:
            glove = np.load(params['embedding'])
            embedding = tf.Variable(glove, trainable = params['embedding_trainable'], name = 'embedding', dtype = tf.float32)

        tf.summary.histogram(embedding.name + '/value', embedding)
        return tf.nn.embedding_lookup(embedding, inputs)

    def conv_op(embd_inp, params):
        fltr = tf.get_variable(
                'conv_fltr', 
                params['kernel'], 
                params['dtype'], 
                regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )

        convout = tf.nn.conv1d(embd_inp, fltr, params['stride'], params['conv_pad'])
        return convout

    def ffn_op(x, params):
        out = x
        if params['ffn_size'] == None:
            ffn_size = []
        else:
            ffn_size = params['ffn_size']
        for unit_size in ffn_size[:-1]:
            out = tf.layers.dense(
                    out, 
                    unit_size, 
                    activation = tf.tanh, 
                    use_bias = True, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                    )
        return tf.layers.dense(
                out, 
                params['label_size'], 
                activation = None, 
                use_bias = True, 
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )

    inputs = features['x']
    
    # raw input to embedded input of shape [batch, length, hidden_size]
    embd_inp = embed_op(inputs, params)
    if params['hidden_size'] != embd_inp.get_shape().as_list()[-1]:
        x = tf.layers.dense(
                embd_inp, 
                params['hidden_size'], 
                activation = None, 
                use_bias = False, 
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
                )
    else:
        x = embd_inp

    # attention bias computation
    padding = ca.embedding_to_padding(x)
    self_attention_bias = ca.attention_bias_ignore_padding(padding)

    
    logits = vector
    # predictions, loss and eval_metric
    predictions = tf.argmax(softmax_out, axis = -1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {'sentiment' : predictions})
    labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, params['label_size'])

    loss_ce = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = logits)
    eval_metric_ops = {
                'accuracy' : tf.metrics.accuracy(tf.argmax(labels, -1), predictions = predictions),
                'pearson_all' : tf.contrib.metrics.streaming_pearson_correlation(softmax_out, tf.cast(labels, tf.float32)),
                'pearson_some' : tf.contrib.metrics.streaming_pearson_correlation(tf.cast(predictions, tf.float32), tf.cast(tf.argmax(labels, -1), tf.float32))
                }

    # regularizaiton loss
    loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    reg_const = params['regularization']  # Choose an appropriate one.
    
    loss = loss_ce + reg_const * loss_reg

    tf.summary.scalar('loss_ce', loss_ce)
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss', loss)
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    learning_rate = params['learning_rate']
    learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), 500, params['decay'], staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.AdamOptimizer()

    #train_op = optimizer.minimize(
        #loss=loss, global_step=tf.train.get_global_step())
    grad_and_var = optimizer.compute_gradients(loss, tf.trainable_variables())
    
    # add histogram summary for gradient
    for grad, var in grad_and_var:
        tf.summary.histogram(var.name + '/gradient', grad)
    train_op = optimizer.apply_gradients(grad_and_var, global_step = tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
