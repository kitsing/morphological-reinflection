import prepare_sigmorphon_data
import common

def main():

    langs = ['turkish', 'navajo', 'german', 'arabic', 'finnish', 'georgian', 'spanish', 'russian']
    for lang in langs:
        train_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/{0}-task1-train'.format(lang)
        dev_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/{0}-task1-dev'.format(lang)
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
if __name__ == '__main__':
    main()