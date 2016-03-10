"""Trains and evaluates a joint-model for inflection generation, using the sigmorphon 2016 shared task data files
and evaluation script.

Usage:
  pycnn_joint_inflection.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--epochs=EPOCHS]
  [--layers=LAYERS] [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    destination path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --cnn-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --feat-input=FEAT             feature input vector dimension
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM
"""

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import codecs
import os
import copy
import pycnn_factored_inflection
from matplotlib import pyplot as plt
from docopt import docopt
from pycnn import *

# default values
INPUT_DIM = 100
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
CHAR_DROPOUT_PROB = 0
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0001
LEARNING_RATE = 0.001  # 0.1

NULL = '%'
UNK = '#'
UNK_FEAT = '@'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim, epochs,
         layers, optimization):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'CHAR_DROPOUT_PROB': CHAR_DROPOUT_PROB, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': REGULARIZATION,
                    'LEARNING_RATE': LEARNING_RATE}

    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # add feature parsing, add features lookup, add features into the decoder
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    feature_alphabet = get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)

    # feat 2 int
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    final_results = {}

    ##################################
    # build model
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, feature_alphabet, feature_types,
                                                                         input_dim, hidden_dim, feat_input_dim, layers)

    # TODO: change so dev will be dev and not test when getting the actual data
    dev_words = test_words
    dev_lemmas = test_lemmas
    dev_feat_dicts = test_feat_dicts

    # train model
    trained_model = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_words, train_lemmas,
                                train_feat_dicts, dev_words, dev_lemmas, dev_feat_dicts, alphabet_index,
                                inverse_alphabet_index, feat_index, feature_types, epochs, optimization,
                                results_file_path)

    # test model
    predictions = predict(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                          inverse_alphabet_index, feat_index, feature_types, test_lemmas,
                          test_feat_dicts)

    accuracy = evaluate_model(predictions, test_lemmas, test_feat_dicts, test_words)

    #####################################
    # get predictions in the same order they appeared in the original file
    # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
    for i, prediction in enumerate(predictions):
        final_results[i] = (test_lemmas[i], test_feat_dicts[i], prediction)

    print 'accuracy: ' + str(accuracy[1])

    write_results_file(hyper_params, accuracy[1], train_path, test_path, results_file_path, sigmorphon_root_dir,
                       final_results)


def write_results_file(hyper_params, accuracy, train_path, test_path, output_file_path, sigmorphon_root_dir,
                       final_results):
    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')

        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')

        f.write('Prediction Accuracy = ' + str(accuracy) + '\n')

    # write predictions in sigmorphon format
    predictions_path = output_file_path + '.predictions'
    with codecs.open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(lines):
                lemma, morph, word = line.split()
                if i in final_results:
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, final_results[i][2]))
                else:
                    # TODO: handle unseen morphs?
                    print u'could not find prediction for {0} {1}'.format(lemma, morph)
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, 'ERROR'))

    # evaluate with sigmorphon script
    evaluation_path = output_file_path + '.evaluation'
    os.chdir(sigmorphon_root_dir)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path +
              ' > ' + evaluation_path)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path)

    print 'wrote results to: ' + output_file_path + '\n' + evaluation_path + '\n' + predictions_path
    return


def build_model(alphabet, feature_alphabet, feature_types, input_dim, hidden_dim, feat_input_dim, layers):
    print 'creating model...'

    model = Model()

    # character embeddings
    model.add_lookup_parameters("char_lookup", (len(alphabet), input_dim))

    # feature embeddings
    # TODO: add another input dim for features?
    model.add_lookup_parameters("feat_lookup", (len(feature_alphabet), feat_input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # TODO: inspect carefully, as dims may be sub-optimal in some cases (many feature types?)
    # 2 * HIDDEN_DIM + 2 * INPUT_DIM + len(feats) * FEAT_INPUT_DIM, as it gets a concatenation of frnn, rrnn
    # (both of HIDDEN_DIM size), previous output char, current lemma char (of INPUT_DIM size) and feats * FEAT_INPUT_DIM
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 2 * input_dim + len(feature_types) * feat_input_dim, hidden_dim,
                              model)

    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


def get_feature_alphabet(feat_dicts):
    feature_alphabet = []
    for f_dict in feat_dicts:
        for f in f_dict:
            feature_alphabet.append(f + ':' + f_dict[f])
    feature_alphabet = list(set(feature_alphabet))
    return feature_alphabet


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def evaluate_model(predictions, lemmas, feat_dicts, words, print_res=False):
    if print_res:
        print 'evaluating model...'

    test_data = zip(lemmas, feat_dicts, words)
    c = 0
    for i, predicted_word in enumerate(predictions):
        (lemma, feats, word) = test_data[i]
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_res:
            print 'lemma: ' + lemma + ' gold: ' + word + ' prediction: ' + predicted_word + ' ' + sign
    accuracy = float(c) / len(predictions)

    if print_res:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predictions)) + '=' + str(accuracy) + \
              '\n\n'

    return len(predictions), accuracy


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_words, train_lemmas, train_feat_dicts,
                dev_words, dev_lemmas, dev_feat_dicts, alphabet_index, inverse_alphabet_index, feat_index,
                feature_types, epochs, optimization, results_file_path):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = AdamTrainer(model, lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = AdagradTrainer(model)
    else:
        trainer = SimpleSGDTrainer(model)

    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_accuracy = -1
    patience = 0
    train_len = len(train_words)
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=epochs).start()
    avg_loss = -1

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word = example
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index,
                                 feat_index, feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                # print 'avg. loss at ' + str(i) + ': ' + str(total_loss / float(i + e*train_len)) + '\n'
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        # TODO: handle when no dev set is available - do best on train set...
        if EARLY_STOPPING:

            if len(dev_lemmas) > 0:

                # get train accuracy
                train_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                            inverse_alphabet_index, feat_index, feature_types, train_lemmas,
                                            train_feat_dicts)

                train_accuracy = evaluate_model(train_predictions, train_lemmas, train_feat_dicts, train_words,
                                                False)[1]

                # get dev accuracy
                dev_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                          inverse_alphabet_index, feat_index, feature_types, dev_lemmas, dev_feat_dicts)

                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words, False)[1]

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    plt.cla()
                    return model

                # get dev loss
                total_dev_loss = 0
                dev_data = zip(dev_lemmas, dev_feat_dicts, dev_words)
                for lemma, feats, word in dev_data:
                    total_dev_loss += one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word,
                                                    alphabet_index, feat_index, feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.2f} dev loss: {2:.2f} accuracy: {3:.2f} best accuracy {4:.2f} \
patience = {5} train accuracy = {6:.2f}'.format(e, avg_loss, avg_dev_loss, dev_accuracy, best_dev_accuracy, patience,
                                                train_accuracy)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                    # return best_model[0]
                    train_progress_bar.finish()
                    plt.cla()
                    return model

                # update lists for plotting
                train_accuracy_y.append(train_accuracy)
                epochs_x.append(e)
                train_loss_y.append(avg_loss)
                dev_loss_y.append(avg_dev_loss)
                dev_accuracy_y.append(dev_accuracy)
            else:
                print 'no dev set for early stopping, running all epochs'

        # finished epoch
        train_progress_bar.update(e)
        with plt.style.context('fivethirtyeight'):
            p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
            p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
            p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
            p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
            plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
        plt.savefig(results_file_path + '_learning_curves.png')
    train_progress_bar.finish()
    plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model


# noinspection PyPep8Naming
def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index, feat_index,
                  feature_types):
    renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK or dropout
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    feats_input = concatenate(feat_vecs)

    # bilstm forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    for c in lemma_char_vecs:
        s = s.add_input(c)
    encoder_frnn_h = s.h()

    # bilstm backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
    encoder_rrnn_h = s.h()

    # concatenate BILSTM final hidden states
    if len(encoder_rrnn_h) == 1 and len(encoder_frnn_h) == 1:
        encoded = concatenate([encoder_frnn_h[0], encoder_rrnn_h[0]])
    else:
        # if there's more than one hidden layer in the rnn's, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    # TODO: change this so it'll be possible to use different dims for input and hidden
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]

    loss = []
    word = BEGIN_WORD + word + END_WORD

    # run the decoder through the sequence and aggregate loss
    for i, word_char in enumerate(word):

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, feats_input])
        s = s.add_input(decoder_input)
        probs = softmax(R * s.output() + bias)
        loss.append(-log(pick(probs, alphabet_index[word_char])))

        # prepare for the next iteration
        prev_output_vec = s.output()

    # TODO: maybe here a "special" loss function is appropriate?
    # loss = esum(loss)
    loss = average(loss)

    return loss


# noinspection PyPep8Naming
def predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feat_dict, alphabet_index,
                       inverse_alphabet_index, feat_index, feature_types):
    renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK or dropout
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feat_dict:
            feat_str = feat + ':' + feat_dict[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    feats_input = concatenate(feat_vecs)

    # bilstm forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    for c in lemma_char_vecs:
        s = s.add_input(c)
    encoder_frnn_h = s.h()

    # bilstm backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
    encoder_rrnn_h = s.h()

    # concatenate BILSTM final hidden states
    if len(encoder_rrnn_h) == 1 and len(encoder_frnn_h) == 1:
        encoded = concatenate([encoder_frnn_h[0], encoder_rrnn_h[0]])
    else:
        # if there's more than one hidden layer in the rnn's, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted = ''

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished or unknown character, pad with epsilon chars
        if i < len(lemma) and lemma[i] in alphabet_index:
                lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]

        # prepare input vector and perform LSTM step
        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, feats_input])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        probs = softmax(R * s.output() + bias)
        probs = probs.vec_value()
        next_predicted_char_index = argmax(probs)
        predicted = predicted + inverse_alphabet_index[next_predicted_char_index]

        # check if reached end of word
        if predicted[-1] == END_WORD:
            break

        # prepare for the next iteration
        prev_output_vec = char_lookup[next_predicted_char_index]
        i += 1

    # remove the begin and end word symbols
    return predicted[1:-1]


def predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, feat_index,
            feature_types, lemmas, feats):
    test_data = zip(lemmas, feats)
    predictions = []
    for lemma, feat_dict in test_data:
        predicted_word = predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feat_dict,
                                            alphabet_index, inverse_alphabet_index, feat_index, feature_types)
        predictions.append(predicted_word)

    return predictions


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path = arguments['TRAIN_PATH']
    else:
        train_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['TEST_PATH']:
        test_path = arguments['TEST_PATH']
    else:
        test_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['RESULTS_PATH']:
        results_file_path = arguments['RESULTS_PATH']
    else:
        results_file_path = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_'\
                            + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    if arguments['--input']:
        input_dim = int(arguments['--input'])
    else:
        input_dim = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim = int(arguments['--hidden'])
    else:
        hidden_dim = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim = int(arguments['--feat-input'])
    else:
        feat_input_dim = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs = int(arguments['--epochs'])
    else:
        epochs = EPOCHS
    if arguments['--layers']:
        layers = int(arguments['--layers'])
    else:
        layers = LAYERS
    if arguments['--optimization']:
        optimization = arguments['--optimization']
    else:
        optimization = OPTIMIZATION

    print arguments

    main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim,
         epochs, layers, optimization)
