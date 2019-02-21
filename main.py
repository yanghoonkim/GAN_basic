import argparse
import numpy as np
import tensorflow as tf

import params
import model as model

FLAGS = None


def main(unused):
    
    # Enable logging for tf.estimator
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # config
    config = tf.contrib.learn.RunConfig(
            model_dir = FLAGS.model_dir, 
            keep_checkpoint_max = 500, 
            save_checkpoints_steps = 100)
    
    # load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # define estimator
    nn = tf.estimator.Estimator(model_fn=model.attn_net, config = config, params=model_params)
    
    # load training data
    train_data = np.load(FLAGS.train_data)
    train_label = np.load(FLAGS.train_label)

    # load lexicon data
    if FLAGS.lexicon_train is not None:
        train_lexicon = np.load(FLAGS.lexicon_train)
        dev_lexicon = np.load(FLAGS.lexicon_dev)
        test_lexicon = np.load(FLAGS.lexicon_test)

    # data shuffling for training data
    permutation = np.random.permutation(len(train_label))
    train_data = train_data[permutation]
    train_label = train_label[permutation]

    if FLAGS.lexicon_train is not None:
        train_lexicon = train_lexicon[permutation]

    # training input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, 'lexicon': train_lexicon},
        y=train_label,
        batch_size = model_params['batch_size'],
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    
    # load evaluation data
    eval_data = np.load(FLAGS.eval_data)
    eval_label = np.load(FLAGS.eval_label)
    
    # evaluation input function for estimator
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data, 'lexicon': dev_lexicon},
        y = eval_label,
        num_epochs=1,
        shuffle=False)  

    # define experiment
    exp_nn = tf.contrib.learn.Experiment(
            estimator = nn, 
            train_input_fn = train_input_fn, 
            eval_input_fn = eval_input_fn,
            train_steps = FLAGS.steps,
            min_eval_frequency = 100)

    # train and evaluate
    if FLAGS.mode == 'train':
        exp_nn.train_and_evaluate()
    
    elif FLAGS.mode == 'eval':
        exp_nn.evaluate(delay_secs = 0)

    else: # 'pred'
        # load preprocessed prediction data
        pred_data = np.load(FLAGS.test_data)

        # prediction input function for estimator
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"x" : pred_data, 'lexicon': test_lexicon},
                shuffle = False
                )

        # prediction
        predict_results = nn.predict(input_fn = pred_input_fn)
        if 'ec_train' in FLAGS.train_data:
            write_sem5(predict_results)
        elif 'voc_train' in FLAGS.train_data:
            write_sem4(predict_results)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'train, eval')
    parser.add_argument('--train_data', type = str, default= '', help = 'path to the training data.')
    parser.add_argument('--train_label', type = str, default = '', help = 'path to the training label.')
    parser.add_argument('--eval_data', type = str, default = '', help = 'path to the evaluation data. ')
    parser.add_argument('--eval_label', type = str, default = '', help = 'path to the evaluation label.')
    parser.add_argument('--test_data', type = str, default = '', help = 'path to the test data')
    parser.add_argument('--test_origin', type = str, default = '')
    parser.add_argument('--lexicon_train', type = str, help = 'path to lexicon data')
    parser.add_argument('--lexicon_dev', type = str, help = 'path to the lexicon data')
    parser.add_argument('--lexicon_test', type = str, help = 'path to the lexicon data')
    parser.add_argument('--model_dir', type = str, help = 'path to save the model')
    parser.add_argument('--pred_dir', type = str, help = 'path to save the predictions')
    parser.add_argument('--prob_dir', type = str, default = 'None', help = 'path to save the predicted probability')
    parser.add_argument('--params', type = str, help = 'parameter setting')
    parser.add_argument('--steps', type = int, default = 200000, help = 'training step size')
    parser.add_argument('--num_epochs', default = 10, help = 'training epoch size')
    FLAGS = parser.parse_args()

    tf.app.run(main)
