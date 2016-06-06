"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  task1_attention.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] [--dev=DEV] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    destination path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
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
"""

import sys
sys.path.insert(0, './machine_translation/')
import prepare_data
import subprocess
import numpy as np
import random
import progressbar
import datetime
import time
import os
from multiprocessing import Pool
from matplotlib import pyplot as plt
import codecs
from docopt import docopt

import prepare_sigmorphon_data
import common


def main(train_path, dev_path, test_path, results_path):
    # read morph input files (train+dev)
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    if dev_path != 'NONE':
        (dev_words, dev_lemmas, dev_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)

        # merge the train and dev files, if dev exists
        train_lemmas += dev_lemmas
        train_words += dev_words
        train_feat_dicts += dev_feat_dicts

    # TODO: optional - implement data augmentation

    # concatenate feats and characters for input
    tokenized_test_inputs, tokenized_test_outputs = convert_to_MED_format(test_feat_dicts, test_lemmas, test_words)
    tokenized_train_inputs, tokenized_train_outputs = convert_to_MED_format(train_feat_dicts, train_lemmas, train_words)

    parallel_data = zip(tokenized_train_inputs, tokenized_train_outputs)

    # write input and output files
    train_inputs_file_path, train_outputs_file_path = write_converted_file(results_path,
                                                               tokenized_train_inputs,
                                                               tokenized_train_outputs,
                                                               'train.in',
                                                               'train.out')

    train_inputs_file_path, train_outputs_file_path = write_converted_file(results_path,
                                                                           tokenized_train_inputs,
                                                                           tokenized_train_outputs,
                                                                           'train.in.tok',
                                                                           'train.out.tok')

    test_inputs_file_path, test_outputs_file_path = write_converted_file(results_path,
                                                                           tokenized_test_inputs,
                                                                           tokenized_test_outputs,
                                                                           'test.in',
                                                                           'test.out')

    test_inputs_file_path, test_outputs_file_path = write_converted_file(results_path,
                                                                         tokenized_test_inputs,
                                                                         tokenized_test_outputs,
                                                                         'test.in.tok',
                                                                         'test.out.tok')

    # HARD preprocess by instantiating the args variables in prepare_data.py to point the created files
    # only changes in original preprocess.py code are:
    # args.source = 'train.in'
    # args.target = 'train.out'
    # args.source_dev = 'test.in'
    # args.target_dev = 'test.out'
    # tr_files = ['/Users/roeeaharoni/GitHub/morphological-reinflection/src/machine_translation/data/train.in',
    #             '/Users/roeeaharoni/GitHub/morphological-reinflection/src/machine_translation/data/train.out']
    # and change shuf to gshuf on mac

    # preprocess using the MILA scripts - create_vocabularies(), shuffle()
    # train_files = [train_inputs_file_path, train_outputs_file_path]
    # test_files = [test_inputs_file_path, test_outputs_file_path]
    # preprocess_file = './machine_translation/preprocess.py'
    # OUTPUT_DIR = './data/'

    # Apply preprocessing and construct vocabularies
    # src_filename, trg_filename = create_vocabularies(tr_files, preprocess_file, OUTPUT_DIR)

    # Shuffle datasets
    # prepare_data.shuffle_parallel(os.path.join(OUTPUT_DIR, src_filename),
    #                                                       os.path.join(OUTPUT_DIR, trg_filename))

    # run training script on the preprocessed files

    return


def write_converted_file(results_path, tokenized_train_inputs, tokenized_train_outputs, in_suffix, out_suffix):
    inputs_file_path = results_path + in_suffix
    with codecs.open(inputs_file_path, 'w', encoding='utf8') as inputs:
        for input in tokenized_train_inputs:
            inputs.write(u'{}\n'.format(input))
    outputs_file_path = results_path + out_suffix
    with codecs.open(outputs_file_path, 'w', encoding='utf8') as outputs:
        for output in tokenized_train_outputs:
            outputs.write(u'{}\n'.format(output))
    print 'created source file: {} \n and target file:{}\n'.format(inputs_file_path, outputs_file_path)
    return inputs_file_path, outputs_file_path


def convert_to_MED_format(train_feat_dicts, train_lemmas, train_words):
    tokenized_inputs = []
    tokenized_outputs = []
    train_set = zip(train_lemmas, train_feat_dicts, train_words)
    for i, example in enumerate(train_set):
        lemma, feats, word = example
        concatenated_input = ''
        for feat in feats:
            concatenated_input += feat + '=' + feats[feat] + ' '
        for char in lemma:
            concatenated_input += char + ' '

        # remove redundant space in the end
        tokenized_inputs.append(concatenated_input[:-1])

        # tokenize output
        tokenized_output = ''
        for char in word:
            tokenized_output += char + ' '

        # remove redundant space in the end
        tokenized_outputs.append(tokenized_output[:-1])
    return tokenized_inputs, tokenized_outputs


def create_vocabularies(tr_files, preprocess_file, output_dir):
    source = 'inputs.txt'
    target = 'outputs.txt'
    source_vocab = 999

    src_vocab_name = os.path.join(
        output_dir, 'vocab.{}-{}.{}.pkl'.format(
            source, target, source))

    trg_vocab_name = os.path.join(
        output_dir, 'vocab.{}-{}.{}.pkl'.format(
            source, target, target))


    src_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(source)][0]]) + '.tok'

    trg_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(target)][0]]) + '.tok'


    print "Creating source vocabulary [{}]".format(src_vocab_name)
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, source_vocab,
            os.path.join(output_dir, src_filename)),
            shell=True)

    print "Creating target vocabulary [{}]".format(trg_vocab_name)
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)

    return src_filename, trg_filename

if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['--dev']:
        dev_path_param = arguments['dev']
    else:
        dev_path_param = 'NONE'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = 'NONE'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    main(train_path_param, dev_path_param, test_path_param, results_file_path_param)