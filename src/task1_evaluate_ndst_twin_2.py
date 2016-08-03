"""Trains and evaluates a joint-structured-model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  evaluate_best_nfst_models.py [--cnn-mem MEM][--input=INPUT] [--feat-input=FEAT][--hidden=HIDDEN]
  [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--ensemble=ENSEMBLE] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    train data path
  TEST_PATH     test data path
  RESULTS_PATH  results file to load the models from
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
  --ensemble=ENSEMBLE           ensemble model paths, separated by comma

"""

import time
from collections import defaultdict

import docopt
import sys

import task1_ndst_twin_2
import prepare_sigmorphon_data
import datetime
import common
import traceback
from pycnn import *

# default values
INPUT_DIM = 150
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 150
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'
STEP = '^'
ALIGN_SYMBOL = '~'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, ensemble):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization}


    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train and test data
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

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(3 * MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # indicates the FST to step forward in the input
    alphabet.append(STEP)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    # cluster the data by POS type (features)
    train_cluster_to_data_indices = common.cluster_data_by_pos(train_feat_dicts)
    test_cluster_to_data_indices = common.cluster_data_by_pos(test_feat_dicts)

    # cluster the data by inflection type (features)
    # train_cluster_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
    # test_cluster_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)

    accuracies = []
    final_results = {}

    # factored model: new model per inflection type
    for cluster_index, cluster_type in enumerate(train_cluster_to_data_indices):

        # get the inflection-specific data
        train_cluster_words = [train_words[i] for i in train_cluster_to_data_indices[cluster_type]]
        if len(train_cluster_words) < 1:
            print 'only {} samples for this inflection type. skipping'.format(str(len(train_cluster_words)))
            continue
        else:
            print 'now evaluating model for cluster ' + str(cluster_index + 1) + '/' + \
                  str(len(train_cluster_to_data_indices)) + ': ' + cluster_type + ' with ' + \
                  str(len(train_cluster_words)) + ' examples'

        # test best model
        try:
            test_cluster_lemmas = [test_lemmas[i] for i in test_cluster_to_data_indices[cluster_type]]
            test_cluster_words = [test_words[i] for i in test_cluster_to_data_indices[cluster_type]]
            test_cluster_feat_dicts = [test_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]

            if ensemble:
                # load ensemble models
                ensemble_model_names = ensemble.split(',')
                print 'ensemble paths:\n'
                print '\n'.join(ensemble_model_names)
                ensemble_models = []
                for ens in ensemble_model_names:
                    model, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(str(cluster_index),
                                                                                     alphabet,
                                                                                     ens,
                                                                                     input_dim,
                                                                                     hidden_dim,
                                                                                     layers,
                                                                                     feature_alphabet,
                                                                                     feat_input_dim,
                                                                                     feature_types)

                    ensemble_models.append((model, encoder_frnn, encoder_rrnn, decoder_rnn))


                # predict the entire test set with each model in the ensemble
                ensemble_predictions = []
                for em in ensemble_models:
                    model, encoder_frnn, encoder_rrnn, decoder_rnn = em
                    predicted_templates = task1_ndst_twin_2.predict_templates(model, decoder_rnn,
                                                                                 encoder_frnn,
                                                                                 encoder_rrnn,
                                                                                 alphabet_index,
                                                                                 inverse_alphabet_index,
                                                                                 test_cluster_lemmas,
                                                                                 test_cluster_feat_dicts,
                                                                                 feat_index,
                                                                                 feature_types)
                    ensemble_predictions.append(predicted_templates)

                predicted_templates = {}
                string_to_template = {}

                # perform voting for each test input - joint_index is a lemma+feats representation
                test_data = zip(test_cluster_lemmas, test_cluster_feat_dicts, test_cluster_words)
                for i, (lemma, feat_dict, word) in enumerate(test_data):
                    joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
                    prediction_counter = defaultdict(int)
                    for ens in ensemble_predictions:
                        prediction_str = ''.join(task1_ndst_twin_2.instantiate_template(ens[joint_index], lemma))
                        prediction_counter[prediction_str] = prediction_counter[prediction_str] + 1
                        string_to_template[prediction_str] = ens[joint_index]
                        print 'template: {} prediction: {}'.format(prediction_str)

                    # return the most predicted output
                    predicted_template_string = max(prediction_counter, key=prediction_counter.get)
                    print u'chosen:{} with {} votes\n'.format(predicted_template_string,
                                                              prediction_counter[predicted_template_string])
                    predicted_templates[joint_index] = string_to_template[predicted_template_string]

                    # progress indication
                    sys.stdout.write("\r%d%%" % (float(i) / len(test_cluster_lemmas) * 100))
                    sys.stdout.flush()
                ##

            else:
                # load best model - no ensemble
                best_model, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(str(cluster_index), alphabet,
                                                                                  results_file_path, input_dim,
                                                                                  hidden_dim, layers,
                                                                                  feature_alphabet, feat_input_dim,
                                                                                  feature_types)
                print 'starting to predict for cluster: {}'.format(cluster_type)
                try:
                    predicted_templates = task1_ndst_twin_2.predict_templates(best_model,
                                                                      decoder_rnn,
                                                                      encoder_frnn,
                                                                      encoder_rrnn,
                                                                      alphabet_index,
                                                                      inverse_alphabet_index,
                                                                      test_cluster_lemmas,
                                                                      test_cluster_feat_dicts,
                                                                      feat_index,
                                                                      feature_types)
                except Exception as e:
                    print e
                    traceback.print_exc()

            print 'evaluating predictions for cluster: {}'.format(cluster_type)
            try:
                accuracy = task1_ndst_twin_2.evaluate_model(predicted_templates,
                                                        test_cluster_lemmas,
                                                        test_cluster_feat_dicts,
                                                        test_cluster_words,
                                                        feature_types,
                                                        print_results=True)
            except Exception as e:
                print e
                traceback.print_exc()
            accuracies.append(accuracy)

            # get predicted_templates in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_cluster_to_data_indices[cluster_type]:
                joint_index = test_lemmas[i] + ':' + common.get_morph_string(test_feat_dicts[i], feature_types)
                inflection = task1_ndst_twin_2.instantiate_template(predicted_templates[joint_index],
                                                                    test_lemmas[i])

                final_results[i] = (test_lemmas[i], test_feat_dicts[i], inflection)

        except KeyError:
            print 'could not find relevant examples in test data for cluster: ' + cluster_type
            print 'clusters in test are: {}'.format(test_cluster_to_data_indices.keys())
            print 'clusters in train are: {}'.format(train_cluster_to_data_indices.keys())

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals) / len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0] * accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom / mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    if 'test' in test_path:
        suffix = '.best.test'
    else:
        suffix = '.best'
    common.write_results_file(hyper_params, micro_average_accuracy, train_path,
                                              test_path, results_file_path + suffix, sigmorphon_root_dir,
                                              final_results)


def load_best_model(morph_index, alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):

    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    model, encoder_frnn, encoder_rrnn, decoder_rnn = task1_ndst_twin_2.build_model(alphabet, input_dim, hidden_dim,
                                                                                        layers, feature_types,
                                                                                        feat_input_dim,
                                                                                        feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, encoder_frnn, encoder_rrnn, decoder_rnn


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

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
    if arguments['--ensemble']:
        ensemble = arguments['--ensemble']
    else:
        ensemble = False

    print arguments

    main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, ensemble)
