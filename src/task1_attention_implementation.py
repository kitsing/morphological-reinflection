"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

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

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import common
import pycnn as pc

from matplotlib import pyplot as plt
from docopt import docopt

# default values
INPUT_DIM = 300
FEAT_INPUT_DIM = 300
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADADELTA'
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

    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

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
    if os.path.isfile(model_file_name) and not override:
        print 'loading existing model from {}'.format(model_file_name)
        initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(alphabet, results_file_path, input_dim,
                                                                                 hidden_dim, layers, feature_alphabet,
                                                                                 feat_input_dim, feature_types)
        print 'loaded existing model successfully'
    else:
        print 'could not find existing model or explicit override was requested. starting training from scratch...'
        initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                             feature_types, feat_input_dim,
                                                                             feature_alphabet)

    trained_model, last_epoch, best_epoch = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn,
                                                        train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                                                        dev_feat_dicts, dev_words, alphabet_index,
                                                        inverse_alphabet_index,
                                                        epochs, optimization, results_file_path, feat_index,
                                                        feature_types, plot)

    print 'last epoch is {}'.format(last_epoch)
    print 'best epoch is {}'.format(best_epoch)
    print 'finished training'

    # evaluate last model on test
    predicted_templates = predict_templates(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                            inverse_alphabet_index, test_lemmas, test_feat_dicts, feat_index,
                                            feature_types)
    if len(predicted_templates) > 0:
        evaluate_model(predicted_templates, test_lemmas, test_feat_dicts, test_words, feature_types,
                       print_results=False)

    # evaluate best models
    os.system('python task1_evaluate_best_joint_structured_models_blstm_feed_fix.py --cnn-mem 6096 --input={0} \
    --hidden={1} --feat-input={2} --epochs={3} --layers={4} --optimization={5} {6} {7} {8} {9}'.format(
                                                                                                input_dim,
                                                                                                hidden_dim,
                                                                                                feat_input_dim,
                                                                                                epochs,
                                                                                                layers,
                                                                                                optimization,
                                                                                                train_path,
                                                                                                test_path,
                                                                                                results_file_path,
                                                                                                sigmorphon_root_dir))
    return


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def load_best_model(alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):

    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                 feature_types, feat_input_dim, feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, encoder_frnn, encoder_rrnn, decoder_rnn


# noinspection PyUnusedLocal
def build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim, feature_alphabet):
    print 'creating model...'

    model = pc.Model()

    # character embeddings
    model.add_lookup_parameters("char_lookup", (len(alphabet), input_dim))

    # feature embeddings
    model.add_lookup_parameters("feat_lookup", (len(feature_alphabet), feat_input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), 3 * hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)

    # attention MLPs

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    model.add_parameters("W_a", 3 * hidden_dim)

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    model.add_parameters("W_c", (3 * hidden_dim, 3 * hidden_dim))

    # 1 * HIDDEN_DIM - gets only the feedback input
    decoder_rnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                dev_feat_dicts, dev_words, alphabet_index, inverse_alphabet_index, epochs, optimization,
                results_file_path, feat_index, feature_types, plot):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = pc.AdamTrainer(model, lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = pc.MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = pc.SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = pc.AdagradTrainer(model)
    elif optimization == 'ADADELTA':
        trainer = pc.AdadeltaTrainer(model)
    else:
        trainer = pc.SimpleSGDTrainer(model)

    train_sanity_set_size = 100
    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_accuracy = -1
    best_train_accuracy = -1
    best_dev_epoch = 0
    best_train_epoch = 0
    patience = 0
    train_len = len(train_words)
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1
    e = 0

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word = example
            loss = compute_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word,
                                alphabet_index, feat_index, feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

            if i % 100 == 0 and i > 0:
                print 'went through {} examples out of {}'.format(i, train_len)

        if EARLY_STOPPING:
            print 'starting epoch evaluation'

            # get train accuracy
            print 'train sanity prediction:'
            train_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                  inverse_alphabet_index, train_lemmas[:train_sanity_set_size],
                                                  train_feat_dicts[:train_sanity_set_size],
                                                  feat_index,
                                                  feature_types)
            print 'train sanity evaluation:'
            train_accuracy = evaluate_model(train_predictions, train_lemmas[:train_sanity_set_size],
                                            train_feat_dicts[:train_sanity_set_size],
                                            train_words[:train_sanity_set_size],
                                            feature_types, True)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                best_train_epoch = e

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_lemmas) > 0:
                print 'dev prediction:'
                # get dev accuracy
                dev_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                    inverse_alphabet_index, dev_lemmas, dev_feat_dicts, feat_index,
                                                    feature_types)
                print 'dev evaluation:'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words, feature_types,
                                              print_results=False)[1]

                if dev_accuracy >= best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    best_dev_epoch = e

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

                # get dev loss
                total_dev_loss = 0
                for i in xrange(len(dev_lemmas)):
                    total_dev_loss += compute_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, dev_lemmas[i],
                                                   dev_feat_dicts[i], dev_words[i], alphabet_index, feat_index,
                                                   feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} train accuracy = {4:.4f} \
 best dev accuracy {5:.4f} (epoch {8}) best train accuracy: {6:.4f} (epoch {9}) patience = {7}'.format(
                                                                                                e,
                                                                                                avg_loss,
                                                                                                avg_dev_loss,
                                                                                                dev_accuracy,
                                                                                                train_accuracy,
                                                                                                best_dev_accuracy,
                                                                                                best_train_accuracy,
                                                                                                patience,
                                                                                                best_dev_epoch,
                                                                                                best_train_epoch)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                    # return best_model[0]
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e
            else:

                # if no dev set is present, optimize on train set
                print 'no dev set for early stopping, running all epochs until perfectly fitting or patience was \
                reached on the train set'

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                print 'epoch: {0} train loss: {1:.4f} train accuracy = {2:.4f} best train accuracy: {3:.4f} \
                patience = {4}'.format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                # found "perfect" model on train set or patience has reached
                if train_accuracy == 1 or patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)

        if plot:
            with plt.style.context('fivethirtyeight'):
                p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
                p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
                plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
            plt.savefig(results_file_path + 'plot.png')

    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: {} best epoch on dev: {} best epoch on train: {}'.format(str(avg_loss),
                                                                                                     best_dev_epoch,
                                                                                                     best_train_epoch)
    return model, e, best_train_epoch


# noinspection PyPep8Naming
def compute_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index, feat_index,
                 feature_types):
    pc.renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = pc.parameter(model["R"])
    bias = pc.parameter(model["bias"])
    W_a = pc.parameter(model["W_a"])
    W_c = pc.parameter(model["W_c"])

    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index,
                                           feat_lookup, feats, feature_types, lemma)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    loss = []
    padded_word = word + END_WORD

    # run the decoder through the output sequence and aggregate loss
    for i, output_char in enumerate(padded_word):

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        attention_output_vector, alphas = attend(blstm_outputs, decoder_rnn_output, R, bias, W_c, W_a)

        current_loss = pc.pickneglogsoftmax(attention_output_vector, alphabet_index[output_char])
        # print 'computed readout layer'
        loss.append(current_loss)

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[alphabet_index[output_char]]

    total_sequence_loss = pc.esum(loss)
    # loss = average(loss)

    return total_sequence_loss


def bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs):

    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in lemma_char_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    for i in xrange(len(lemma_char_vecs)):
        blstm_outputs.append(pc.concatenate([frnn_outputs[i], rrnn_outputs[len(lemma_char_vecs) - i - 1]]))

    return blstm_outputs


# noinspection PyPep8Naming
def predict_output_sequence(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, alphabet_index,
                            inverse_alphabet_index, feat_index, feature_types):
    pc.renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = pc.parameter(model["R"])
    bias = pc.parameter(model["bias"])
    W_a = pc.parameter(model["W_a"])
    W_c = pc.parameter(model["W_c"])

    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index,
                                           feat_lookup, feats, feature_types, lemma)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted_sequence = []

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        # perform attention step
        attention_output_vector, alphas = attend(blstm_outputs, decoder_rnn_output, R, bias, W_c, W_a)

        # find best candidate output
        probs = pc.softmax(attention_output_vector)
        next_char_index = common.argmax(probs.vec_value())
        predicted_sequence.append(inverse_alphabet_index[next_char_index])

        # check if reached end of word
        if predicted_sequence[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[next_char_index]
        i += 1

    # remove the end word symbol
    return predicted_sequence[0:-1]


def encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats,
                           feature_types, lemma):

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = [char_lookup[alphabet_index[BEGIN_WORD]]]
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                continue
                # handle UNK or dropout
                # feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
                # else:
                # feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    # feats_input = pc.concatenate(feat_vecs)

    # convert characters to matching embeddings, if UNK handle properly
    # padded_lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    lemma_char_vecs.append(char_lookup[alphabet_index[END_WORD]])

    # add feats in the beginning of the input sequence
    feats_and_lemma_vecs = feat_vecs + lemma_char_vecs

    # create bidirectional representation
    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, feats_and_lemma_vecs)
    return blstm_outputs


def attend(blstm_outputs, h_t, R, bias, W_c, W_a):
    # attention mechanism:
    # iterate through input states to compute alphas
    # print 'computing scores...'
    scores = [pc.transpose(W_a) * pc.concatenate([h_t, h_input]) for h_input in blstm_outputs]
    # print 'computed scores'
    # normalize to alphas using softmax
    # print 'computing alphas...'
    alphas = pc.softmax(pc.concatenate(scores))
    # print 'computed alphas...'
    # compute c using alphas
    # print 'computing c...'
    c = pc.esum([blstm_outputs[j] * pc.scalarInput(alpha) for j, alpha in enumerate(alphas.vec_value())])
    # print 'computed c'
    # print 'c len is {}'.format(len(c.vec_value()))
    # compute output state h~ using c and the decoder's h (global attention variation from Loung and Manning 2015)
    # print 'computing h~...'
    h_output = pc.tanh(W_c * pc.concatenate([h_t, c]))
    # print 'len of h_output is {}'.format(len(h_output.vec_value()))
    # print 'computed h~'
    # compute output probabilities
    # print 'computing readout layer...'
    attention_output_vector = R * h_output + bias
    return attention_output_vector, alphas


def predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, lemmas,
                      feats, feat_index, feature_types):
    print 'predicting...'
    predictions = {}
    data_len = len(lemmas)
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
        predicted_template = predict_output_sequence(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma,
                                                     feat_dict, alphabet_index, inverse_alphabet_index, feat_index,
                                                     feature_types)
        if i % 100 == 0 and i > 0:
            print 'predicted {} examples out of {}'.format(i, data_len)

        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_template

    return predictions


def evaluate_model(predicted_templates, lemmas, feature_dicts, words, feature_types, print_results=True):
    if print_results:
        print 'evaluating model...'

    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predicted_word = ''.join(predicted_templates[joint_index])
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_results:
            print u'lemma: {} gold: {} template: {} prediction: {} correct: {}'.format(
                                                                            lemma, words[i],
                                                                            ''.join(predicted_templates[joint_index]),
                                                                            predicted_word,
                                                                            sign)
    accuracy = float(c) / len(predicted_templates)

    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_templates)) + '=' + \
              str(accuracy) + '\n\n'

    return len(predicted_templates), accuracy


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
