import sys
import os
import time
import datetime

def main():

    sig_root = '/Users/roeeaharoni/research_data/sigmorphon2016-master'
    results_dir = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results'
    #langs = ['german', 'turkish', 'spanish', 'navajo']#,
    langs = ['arabic', 'finnish', 'georgian', 'russian']
    print 'now training langs: ' + str(langs)
    for lang in langs:
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H:%M:%S')
        os.chdir('/Users/roeeaharoni/Github/morphological-reinflection/src')
        os.system('python pycnn_faruqui_inflection.py --cnn-mem 4096 --input=100 --hidden=100 --epochs=100 --layers=2 \
                  --optimization ADAM \
                  {0}/data/{1}-task1-train \
                  {0}/data/{1}-task1-dev \
                  {3}/{1}_{2}_results.txt \
                  {0}'.format(sig_root, lang, st, results_dir))
        end = time.time()
        print 'finished ' + lang + ' in ' + str(end - start)

        #evaluate_baseline(lang, results_dir, sig_root)


def evaluate_baseline(lang, results_dir, sig_root):
    os.chdir(sig_root + '/src/baseline')

    # run baseline system
    os.system('./baseline.py --task=1 --language={0} \
        --path={1}/data/ > {2}/baseline_{0}_task1_predictions.txt'.format(lang, sig_root, results_dir))
    os.chdir(sig_root + '/src')

    # eval baseline system
    os.system('python evalm.py --gold ../data/{0}-task1-dev --guesses \
        {1}/baseline_{0}_task1_predictions.txt'.format(lang, results_dir))


if __name__ == '__main__':
    main()