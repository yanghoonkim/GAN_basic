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
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_label = mnist.train.labels


    # data shuffling for training data
    permutation = np.random.permutation(len(train_label))
    train_data = train_data[permutation]
    train_label = train_label[permutation]


    # training input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_label,
        batch_size = model_params['batch_size'],
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    
    # load evaluation data
    eval_data = mnist.validation.images
    eval_label = mnist.validation.labels
    
    # evaluation input function for estimator
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
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
        pred_data = mnist.text.images

        # prediction input function for estimator
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"x" : pred_data},
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
    parser.add_argument('--model_dir', type = str, help = 'path to save the model')
    parser.add_argument('--pred_dir', type = str, help = 'path to save the predictions')
    parser.add_argument('--prob_dir', type = str, default = 'None', help = 'path to save the predicted probability')
    parser.add_argument('--params', type = str, help = 'parameter setting')
    parser.add_argument('--steps', type = int, default = 200000, help = 'training step size')
    parser.add_argument('--num_epochs', default = 10, help = 'training epoch size')
    FLAGS = parser.parse_args()

    tf.app.run(main)
