"""Trains and evaluates a factored-model for inflection generation, using the sigmorphon 2016 shared task data files and
evaluation script.

Usage:
  pycnn_faruqui_inflection.py [--input=INPUT] [--hidden=HIDDEN] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    destination path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --input=INPUT                 input vector dimensions
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
from docopt import docopt
from pycnn import *

# default values
INPUT_DIM = 100
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
CHAR_DROPOUT_PROB = 0
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'

# TODO: try naive substring approach - LCS (Ahlberg 2015)?
# TODO: try running on GPU
# TODO: write evaluation code with sigmorphon script
# TODO: consider different begin, end chars for lemma and word
# TODO: consider different lookup table for lemma and word
# TODO: implement (character?) dropout
# TODO: make different input and hidden dims work
# TODO: plot learning curve
# TODO: implement smart stopping
# TODO: try different learning algorithms (ADAGRAD, ADAM...)
# TODO: refactor so less code repetition in train, predict (both have blstm etc.)
# TODO: think how to give more emphasis on suffix generalization/learning
# TODO: handle unk chars better

def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'CHAR_DROPOUT_PROB': CHAR_DROPOUT_PROB, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization}

    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load data
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feats = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

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

    # cluster the data by inflection type (features)
    train_morph_to_data_indices = get_distinct_morph_types(train_feat_dicts, feats)
    test_morph_to_data_indices = get_distinct_morph_types(test_feat_dicts, feats)

    # debug prints
    # print_data_stats(alphabet, feats, morph_to_data_indices, test_morph_to_data_indices, train_feat_dicts,
    # train_lemmas, train_words)
    accuracies = []
    models = {}
    final_results = {}

    # factored model: new model per inflection type
    for morph_index, morph_type in enumerate(train_morph_to_data_indices):

        # get the inflection-specific data
        train_morph_words = [train_words[i] for i in train_morph_to_data_indices[morph_type]]
        train_morph_lemmas = [train_lemmas[i] for i in train_morph_to_data_indices[morph_type]]
        if len(train_morph_words) < 1:
            print 'only ' + str(len(train_morph_words)) + ' samples for this inflection type. skipping'
            continue
        else:
            print 'now training model for morph ' + str(morph_index) + '/' + str(len(train_morph_to_data_indices)) + \
                  ': ' + morph_type + ' with ' + str(len(train_morph_words)) + ' examples'

        # build model
        initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers)

        # train model
        trained_model = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_morph_words,
                                    train_morph_lemmas, alphabet_index, epochs, optimization)

        # save model
        models[morph_type] = (trained_model, encoder_frnn, encoder_rrnn, decoder_rnn)

        # test model
        try:
            test_morph_lemmas = [test_lemmas[i] for i in test_morph_to_data_indices[morph_type]]
            test_morph_words = [test_words[i] for i in test_morph_to_data_indices[morph_type]]

            predictions = predict(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                             inverse_alphabet_index, test_morph_lemmas, test_morph_words)

            test_data = zip(test_morph_lemmas, test_morph_words)
            accuracy = evaluate_model(predictions, test_data)
            accuracies.append(accuracy)

            # get predictions in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_morph_to_data_indices[morph_type]:
                final_results[i] = (test_lemmas[i], predictions[test_lemmas[i]], morph_type)

        except KeyError:
            print 'could not find relevant examples in test data for morph: ' + morph_type

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals)/len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0]*accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom/mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    write_results_file(hyper_params, macro_avg_accuracy, micro_average_accuracy, train_path, test_path,
                       results_file_path, sigmorphon_root_dir, final_results)


def get_distinct_morph_types(feat_dicts, feats):
    morphs_to_indices = {}
    for i, d in enumerate(feat_dicts):
        s = ''
        for f in sorted(feats):
            if f in d:
                s += f + '=' + d[f] + ':'
            else:
                s += f + '=' + NULL + ':'
        s = s[:-1]
        if s in morphs_to_indices:
            morphs_to_indices[s].append(i)
        else:
            morphs_to_indices[s] = [i]
    return morphs_to_indices


def build_model(alphabet, input_dim, hidden_dim, layers):

    print 'creating model...'

    model = Model()

    # character embeddings
    model.add_lookup_parameters("lookup", (len(alphabet), input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # 3 * HIDDEN_DIM + INPUT_DIM, as it gets a concatenation of frnn, rrnn, previous output char, current lemma char
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 2 * input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


# noinspection PyPep8Naming
def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, word, alphabet_index):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings (or dropout), if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            # char dropout
            drop_char = np.random.choice([True, False], 1, p=[CHAR_DROPOUT_PROB, 1 - CHAR_DROPOUT_PROB])[0]
            if drop_char:
                # TODO: get rid of the exceptions here
                raise KeyError()
            else:
                lemma_char_vecs.append(lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK or dropout
            lemma_char_vecs.append(lookup[alphabet_index[UNK]])

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
    prev_output_vec = lookup[alphabet_index[BEGIN_WORD]]
    loss = []
    word = BEGIN_WORD + word + END_WORD

    # run the decoder through the sequence and aggregate loss
    for i, word_char in enumerate(word):

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec])
        s = s.add_input(decoder_input)
        probs = softmax(R * s.output() + bias)
        loss.append(-log(pick(probs, alphabet_index[word_char])))

        # prepare for the next iteration
        prev_output_vec = s.output()

    # TODO: maybe here a "special" loss function is appropriate?
    loss = esum(loss)

    return loss


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, morph_words, morph_lemmas, alphabet_index, epochs,
                optimization):
    print 'training...'
    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = AdamTrainer(model)
    elif optimization == 'MOMENTUM':
        trainer = MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = AdagradTrainer(model)
    else:
        trainer = SimpleSGDTrainer(model)

    total_loss = 0
    train_len = len(morph_words)

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=epochs).start()
    avg_loss = -1
    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(morph_lemmas, morph_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, example[0], example[1], alphabet_index)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
            # print 'avg. loss at ' + str(i) + ': ' + str(total_loss / float(i + e*train_len)) + '\n'
                avg_loss = total_loss / float(i + e*train_len)
            else:
                avg_loss = total_loss

        train_progress_bar.update(e)

    train_progress_bar.finish()
    print 'finished training. average loss: ' + str(avg_loss)
    return model


def predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, test_lemmas,
            test_words):
    test_data = zip(test_lemmas, test_words)
    predictions = {}
    for lemma, word in test_data:
        predicted_word = predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, alphabet_index,
                                            inverse_alphabet_index)
        predictions[lemma] = predicted_word

    return predictions


# noinspection PyPep8Naming
def predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, alphabet_index, inverse_alphabet_index):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(lookup[alphabet_index[UNK]])

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
        # if there's more than one layer, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted = ''

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished or unknown character, pad with epsilon chars
        if i < len(lemma) and lemma[i] in alphabet_index:
                lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        # prepare input vector and perform LSTM step
        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        probs = softmax(R * s.output() + bias)
        probs = probs.vec_value()
        next_char_index = argmax(probs)
        predicted = predicted + inverse_alphabet_index[next_char_index]

        # check if reached end of word
        if predicted[-1] == END_WORD:
            break

        # prepare for the next iteration
        prev_output_vec = lookup[next_char_index]
        i += 1

    # remove the begin and end word symbols
    return predicted[1:-1]


def evaluate_model(predictions, test_data):

    print 'evaluating model...'

    c = 0
    for i, lemma in enumerate(predictions.keys()):
        (lemma, word) = test_data[i]
        predicted_word = predictions[lemma]
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        print 'lemma: ' + lemma + ' gold: ' + word + ' prediction: ' + predicted_word + ' ' + sign
    accuracy = float(c) / len(predictions)

    print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predictions)) + '=' + str(accuracy) + '\n\n'

    return len(predictions), accuracy


def write_results_file(hyper_params, macro_avg_accuracy, micro_average_accuracy, train_path, test_path,
                       output_file_path, sigmorphon_root_dir, final_results):

    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')

        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')

        f.write('Prediction Accuracy = ' + str(micro_average_accuracy) + '\n')
        f.write('Macro-Average Accuracy = ' + str(macro_avg_accuracy) + '\n')

    # write predictions in sigmorphon format
    predictions_path = output_file_path + '.predictions'
    with codecs.open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(lines):
                lemma, morph, word = line.split()
                if i in final_results:
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, final_results[i][1]))
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


def print_data_stats(alphabet, feats, morph_types, test_morph_types, train_feat_dicts, train_lemmas, train_words):
    print '\nalphabet' + str(sorted([f for f in alphabet if all(ord(c) < 128 for c in f)]))
    print 'features' + str(feats)
    print 'train_words: ' + str(len(train_words)) + ' ' + str(train_words[:10])
    print 'train_lemmas: ' + str(len(train_lemmas)) + ' ' + str(train_lemmas[:10])
    print 'train_feat_dicts: ' + str(len(train_feat_dicts)) + ' ' + str(train_feat_dicts[:1])
    print 'morph types: ' + str(len(morph_types)) + ' ' + str(morph_types.keys()[0])
    print 'verb morph types: ' + str(len([m for m in morph_types if 'pos=V' in m]))
    print 'noun morph types: ' + str(len([m for m in morph_types if 'pos=N' in m]))
    print 'test morph types: ' + str(len(test_morph_types)) + ' ' + str(test_morph_types.keys()[0])
    # for morph in morph_types:
    #    print morph


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

if __name__ == '__main__':
    arguments = docopt(__doc__)
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
        results_file_path = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' \
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

    main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization)
