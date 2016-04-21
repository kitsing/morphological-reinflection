import prepare_sigmorphon_data
import common

def main():

    langs = ['russian', 'georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german', 'hungarian', 'maltese']
    for lang in langs:
        train_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/{0}-task1-train'.format(lang)
        dev_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/{0}-task1-dev'.format(lang)
        (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
        (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)
        alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)
        train_cluster_to_data_indices = common.cluster_data_by_pos(train_feat_dicts)
        test_cluster_to_data_indices = common.cluster_data_by_pos(test_feat_dicts)
        train_morph_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
        test_morph_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)
        train_agg = 0
        for cluster in train_cluster_to_data_indices:
            train_agg += len(train_cluster_to_data_indices[cluster])
            print 'train ' + lang + ' ' + cluster + ' : ' + str(len(train_cluster_to_data_indices[cluster])) + ' examples'

        print 'train ' + lang + ' ' + 'agg' + ' : ' + str(train_agg) + ' examples'
        dev_agg = 0
        for cluster in test_cluster_to_data_indices:
            dev_agg += len(test_cluster_to_data_indices[cluster])
            print 'dev ' + lang + ' ' + cluster + ' : ' + str(len(test_cluster_to_data_indices[cluster])) + ' examples'
        print 'dev ' + lang + ' ' + 'agg' + ' : ' + str(dev_agg) + ' examples'
        print lang + ' train morphs: ' + str(len(train_morph_to_data_indices))
        print lang + ' avg ex. per morph: ' + str(sum([len(l) for l in train_morph_to_data_indices.keys()])/float(len(train_morph_to_data_indices)))
        print lang + ' dev morphs: ' + str(len(test_morph_to_data_indices))
        print lang + ' num features: ' + str(len(feature_types))

        for cluster in train_cluster_to_data_indices:
            print 'train ' + lang + ' ' + cluster + ' : ' + str(
                len(train_cluster_to_data_indices[cluster])) + ' examples'
            train_cluster_words = [train_words[i] for i in train_cluster_to_data_indices[cluster]]
            train_cluster_lemmas = [train_lemmas[i] for i in train_cluster_to_data_indices[cluster]]
            prefix_count, suffix_count, same_count, circumfix_count, other_count, lev_avg = get_morpheme_stats(
                train_cluster_words,
                train_cluster_lemmas)
            print "train {0} {1} {2} &  {3} & {4} & {5} & {6} & {7:.3f}".format(
                lang, cluster, prefix_count, suffix_count, same_count, circumfix_count, other_count, lev_avg)

        prefix_count, suffix_count, same_count, circumfix_count, other_count, lev_avg = get_morpheme_stats(
            train_words,
            train_lemmas)
        print "train {0} {1} {2} &  {3} & {4} & {5} & {6} & {7:.3f}".format(
            lang, 'AGG', prefix_count, suffix_count, same_count, circumfix_count, other_count, lev_avg)

def get_morpheme_stats(train_lemmas, train_words):
    prefix_count = 0
    suffix_count = 0
    other_count = 0
    circumfix_count = 0
    same_count = 0
    lev_sum = 0
    for lemma, word in zip(train_lemmas, train_words):
        if lemma == word:
            same_count+=1
            continue

        if lemma in word:
            if word.replace(lemma, '') + lemma == word:
                prefix_count += 1
            if lemma + word.replace(lemma, '') == word:
                suffix_count += 1
            if lemma + word.replace(lemma, '') != word and word.replace(lemma, '') + lemma != word:
                circumfix_count +=1
        else:
            other_count +=1

        lev_sum += levenshtein(lemma, word)


    return prefix_count, suffix_count, same_count, circumfix_count, other_count, lev_sum/float(len(train_lemmas))

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


if __name__ == '__main__':
    main()

