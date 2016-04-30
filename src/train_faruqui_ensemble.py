import sys
import os

def main():

    # train ensemble
    for i in xrange(1):
        langs = ['russian', 'georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german']
        train_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/data/{0}-task1-train.morphtrans.txt'
        dev_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/data/{0}-task1-dev.morphtrans.txt'

        for lang in langs:
            train_file = train_format.format(lang)
            dev_file = dev_format.format(lang)
            train_alphabet_file = train_file + '.word_alphabet'
            train_morph_file = train_file + '.morph_alphabet'
            model_file = '/Users/roeeaharoni/research_data/morphology/morph_trans_models/model_{0}.txt'.format(
                lang)

            eval_file = "/Users/roeeaharoni/research_data/morphology/morph_trans_models/eval_model_{0}.txt".format(
                lang)

            print train_file
            print dev_file
            print train_alphabet_file
            print train_morph_file
            print model_file
            print eval_file


            os.system("/Users/roeeaharoni/GitHub/morph-trans/bin/train-sep-morph --cnn-mem 9064\
            {0} {1} {2} {3} \
            100 30 1e-5 2 {4}".format(
                train_alphabet_file,
                train_morph_file,
                train_file,
                dev_file,
                model_file))

            # test ensemble
            os.system("~/GitHub/morph-trans/bin/eval-ensemble-sep-morph --cnn-mem 9064\
    {0} {1} {2} {3} > {4}".format(
                train_alphabet_file,
                train_morph_file,
                dev_file,
                model_file,
                eval_file))




if __name__ == '__main__':
    main()