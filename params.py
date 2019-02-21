import tensorflow as tf
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 8680,
        embedding = None,
        embedding_trainable = True,
        label_size = 6,
        multi_label = 1, # if >1, multi-label setting
        hidden_size = 50,
        value_depth = 50,
        bucket_sizes = [10, 20, 30, 40, 50],
        
        # attention network parameters
        num_layers = 1,
        num_heads = 10,
        attn_dropout = 0.1,
        residual_dropout = 0.1,
        relu_dropout = 0.1,
        filter_size = 128,
        
        # convolution parameters
        kernel = [10, 50, 30], # kernel shape for tf.nn.conv1d, [filter_width, in_channels, out_channels]
        stride = 1,
        conv_pad = 'VALID', # 'VALID' or 'SAME'
        
        # fully connected network parameters
        ffn_size = None,
        
        # learning parameters
        batch_size = 20,
        learning_rate = 0.02
        
    )

def other_params():
    hparams = basic_params()
    hparams.voca_size = 190496
    hparams.num_layers = 0
    hparams.embedding = 'data/semeval/processed/glove840b_semeval1_5_vocab300_emo_unlabel.npy'
    hparams.add_hparam('regularization', 0.001)
    hparams.learning_rate = 0.001
    hparams.add_hparam('lexicon_effect', 0.0) # lexicon coefficient
    hparams.add_hparam('decay', 0.4) # learning rate decay factor
    return hparams
