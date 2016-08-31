# -*- coding: utf-8 -*-
"""visualization of the attention weights for inflection generation.

Usage:
  task1_joint_structured_inflection_blstm_feedback_fix.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] [--override] TRAIN_PATH DEV_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    train set path path
  DEV_PATH      development set path
  TEST_PATH     test set path
  RESULTS_PATH  results file path
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --cnn-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --hidden=HIDDEN               hidden layer dimensions
  --feat-input=FEAT             feature input vector dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
  --override                    override the existing model with the same name, if exists
"""

import sys
import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import common
import pycnn as pc
import task1_attention_implementation

from collections import defaultdict
from multiprocessing import Pool
from matplotlib import pyplot as plt
from docopt import docopt

# default values
INPUT_DIM = 300
FEAT_INPUT_DIM = 100
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True
BEAM_WIDTH = 5

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'


def main(train_path, dev_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim,
         epochs, layers, optimization, regularization, learning_rate, plot, override):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate}

    (alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types, initial_model,
     inverse_alphabet_index, dev_words, dev_lemmas, dev_feat_dicts) = init_model(dev_path, feat_input_dim, hidden_dim,
                                                                                 input_dim, layers, results_file_path,
                                                                                 test_path, train_path)

    start = 300
    end = 350
    for lemma, feats in zip(dev_lemmas[start:end], dev_feat_dicts[start:end]):
        plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                                 initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                                 hyper_params, input_dim, layers, results_file_path, test_path, train_path, lemma)

    return
    # get user input word and features
    # feats = {u'pos': u'NN', u'num': u'P', u'gen': u'F', u'poss_per': u'2', u'poss_gen': u'M', u'poss_num': u'P'}
    feats = {u'pos': u'NN', u'num': u'P', u'gen': u'F', u'poss_per': u'2', u'poss_gen': u'M',
             u'poss_num': u'P'}  # u'tense' : u'FUTURE', u'poss_per': u'2', u'poss_gen': u'M', u'poss_num': u'P'}

    user_input = u'כלב'
    plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                             initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                             hyper_params, input_dim, layers, results_file_path, test_path, train_path, user_input)

    feats = {u'pos': u'VB', u'num': u'S', u'gen': u'F', u'per': u'3', u'tense': u'FUTURE', u'binyan': u'PAAL'}
    user_input = u'ישן'
    plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                             initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                             hyper_params, input_dim, layers, results_file_path, test_path, train_path, user_input)

    feats = {u'pos': u'VB', u'num': u'P', u'gen': u'M', u'per': u'3', u'tense': u'FUTURE', u'binyan': u'PAAL'}
    user_input = u'ישן'
    plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                             initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                             hyper_params, input_dim, layers, results_file_path, test_path, train_path, user_input)

    feats = {u'pos': u'VB', u'num': u'P', u'gen': u'F', u'per': u'3', u'tense': u'FUTURE', u'binyan': u'PAAL'}
    user_input = u'ישן'
    plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                             initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                             hyper_params, input_dim, layers, results_file_path, test_path, train_path, user_input)
    print 'Bye!'


def plot_attn_for_inflection(alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types,
                             initial_model, inverse_alphabet_index, dev_path, feat_input_dim, feats, hidden_dim,
                             hyper_params, input_dim, layers, results_file_path, test_path, train_path, user_input):
    # predict
    output_seq, alphas_mtx, input_seq, W = predict_output_sequence(initial_model, encoder_frnn, encoder_rrnn,
                                                                   decoder_rnn, user_input, feats, alphabet_index,
                                                                   inverse_alphabet_index, feat_index, feature_types)
    fig, ax = plt.subplots()
    new = []
    for row in W:
        for column in row:
            new.append(list(row))
    # plot heatmap
    # image = np.array(new)
    # ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    # ax.set_title(u'W_a weights viz')

    image = np.array(alphas_mtx)
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

    # fix x axis ticks density
    ax.xaxis.set_ticks(np.arange(0, len(alphas_mtx[0]), 1))

    # fix y axis ticks density
    ax.yaxis.set_ticks(np.arange(0, len(alphas_mtx), 1))

    # set tick labels to meaningful symbols
    ax.set_xticklabels(list(input_seq))
    ax.set_yticklabels(list(output_seq))

    # set title
    input_word = u''.join(input_seq)
    output_word = u''.join(output_seq)
    ax.set_title(u'attention-based alignment: {} -> {}'.format(user_input, output_word[0:-1]))

    print input_word
    print output_word[0:-1]


def init_model(dev_path, feat_input_dim, hidden_dim, input_dim, layers, results_file_path, test_path, train_path):
    # load train and test data
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    (dev_words, dev_lemmas, dev_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)
    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)
    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)
    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}
    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))
    model_file_name = results_file_path + '_bestmodel.txt'
    # load model and everything else needed for prediction
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = task1_attention_implementation.load_best_model(
                                                                                                    alphabet,
                                                                                                    results_file_path,
                                                                                                    input_dim,
                                                                                                    hidden_dim,
                                                                                                    layers,
                                                                                                    feature_alphabet,
                                                                                                    feat_input_dim,
                                                                                                    feature_types)
    print 'loaded existing model successfully'
    return (alphabet_index, decoder_rnn, encoder_frnn, encoder_rrnn, feat_index, feature_types, initial_model,
            inverse_alphabet_index, dev_words, dev_lemmas, dev_feat_dicts)


# noinspection PyPep8Naming
def predict_output_sequence(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, alphabet_index,
                            inverse_alphabet_index, feat_index, feature_types):
    pc.renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = pc.parameter(model["R"])
    bias = pc.parameter(model["bias"])
    W__a = pc.parameter(model["W__a"])
    U__a = pc.parameter(model["U__a"])
    v__a = pc.parameter(model["v__a"])
    W_c = pc.parameter(model["W_c"])

    blstm_outputs = task1_attention_implementation.encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn,
                                                                          encoder_rrnn, feat_index, feat_lookup,
                                                                          feats, feature_types, lemma)
    feat_list = []
    for feat in sorted(feature_types):
        if feat in feats:
            feat_list.append(feats[feat])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted_sequence = []
    alphas_mtx = []

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        # perform attention step
        attention_output_vector, alphas, W = task1_attention_implementation.attend(blstm_outputs, decoder_rnn_output,
                                                                                   W_c, v__a, W__a, U__a)
        val = alphas.vec_value()
        print 'alphas:'
        print val
        alphas_mtx.append(val)

        # compute output probabilities
        # print 'computing readout layer...'
        readout = R * attention_output_vector + bias
        next_char_index = common.argmax(readout.vec_value())
        predicted_sequence.append(inverse_alphabet_index[next_char_index])

        # check if reached end of word
        if predicted_sequence[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[next_char_index]
        i += 1

    # remove the end word symbol
    return predicted_sequence, alphas_mtx, [BEGIN_WORD] + feat_list + list(lemma) + [END_WORD], W


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['DEV_PATH']:
        dev_path_param = arguments['DEV_PATH']
    else:
        dev_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-test'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    if arguments['--input']:
        input_dim_param = int(arguments['--input'])
    else:
        input_dim_param = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim_param = int(arguments['--hidden'])
    else:
        hidden_dim_param = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim_param = int(arguments['--feat-input'])
    else:
        feat_input_dim_param = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS
    if arguments['--layers']:
        layers_param = int(arguments['--layers'])
    else:
        layers_param = LAYERS
    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION
    if arguments['--reg']:
        regularization_param = float(arguments['--reg'])
    else:
        regularization_param = REGULARIZATION
    if arguments['--learning']:
        learning_rate_param = float(arguments['--learning'])
    else:
        learning_rate_param = LEARNING_RATE
    if arguments['--plot']:
        plot_param = True
    else:
        plot_param = False
    if arguments['--override']:
        override_param = True
    else:
        override_param = False

    print arguments

    main(train_path_param, dev_path_param, test_path_param, results_file_path_param, sigmorphon_root_dir_param,
         input_dim_param, hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param,
         regularization_param, learning_rate_param, plot_param, override_param)
