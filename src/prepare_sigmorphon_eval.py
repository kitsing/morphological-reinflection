# Date created: December 13 2015
# Author: Yonatan Belinkov

# convert predictions from torch to sigmorphon format, for evaluation by formal script

import sys, codecs


def convert(sigmorphon_gold_filename, pred_filename, sigmorphon_pred_filename):

    gold_lines = codecs.open(sigmorphon_gold_filename, encoding='utf-8').readlines()
    pred_lines = codecs.open(pred_filename, encoding='utf-8').readlines()
    assert len(gold_lines) == len(pred_lines), 'gold and pred files do not have same number of lines'
    g = codecs.open(sigmorphon_pred_filename, 'w', encoding='utf-8')
    for i in xrange(len(gold_lines)):
        gold_splt = gold_lines[i].strip().split()
        assert len(gold_splt) == 3, 'bad gold line: ' + gold_lines[i]
        gold_lemma, gold_feats, gold_word = gold_splt
        pred_word = pred_lines[i].strip().replace(' ', '')
        g.write(gold_lemma + '\t' + gold_feats + '\t' + pred_word + '\n')
    g.close()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <sigmorphon gold file> <torch pred file> <sigmorphon pred file>'
