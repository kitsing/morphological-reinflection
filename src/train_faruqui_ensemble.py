import sys
import os

def main():

    # train ensemble
    for i in xrange(5):
        os.system("/Users/roeeaharoni/GitHub/morph-trans/bin/train-sep-morph \
        ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-train.word_alphabet \
        ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-train.morph_alphabet \
        ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-train \
        ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-dev \
        100 100 1e-5 1 /Users/roeeaharoni/research_data/morphology/morph_trans_models/model{0}.txt".format(i+1))

    # test ensemble
    os.system("~/GitHub/morph-trans/bin/eval-ensemble-sep-morph --cnn-mem 4096\
    ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-train.word_alphabet \
    ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-train.morph_alphabet \
    ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_german-task1-dev \
    /Users/roeeaharoni/research_data/morphology/morph_trans_models/model1.txt \
    /Users/roeeaharoni/research_data/morphology/morph_trans_models/model2.txt \
    /Users/roeeaharoni/research_data/morphology/morph_trans_models/model3.txt \
    /Users/roeeaharoni/research_data/morphology/morph_trans_models/model4.txt \
    /Users/roeeaharoni/research_data/morphology/morph_trans_models/model5.txt \
    > ~/Dropbox/phd/research/morphology/inflection_generation/morphtrans_ensemble_output.txt")




if __name__ == '__main__':
    main()