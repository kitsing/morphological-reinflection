import align


NULL = '%'

def cluster_data_by_morph_type(feat_dicts, feature_types):
    morphs_to_indices = {}
    for i, feat_dict in enumerate(feat_dicts):
        s = get_morph_string(feat_dict, feature_types)
        if s in morphs_to_indices:
            morphs_to_indices[s].append(i)
        else:
            morphs_to_indices[s] = [i]
    return morphs_to_indices


def get_morph_string(feat_dict, feature_types):
    s = ''
    for f in sorted(feature_types):
        if f in feat_dict:
            s += f + '=' + feat_dict[f] + ':'
        else:
            s += f + '=' + NULL + ':'
    s = s[:-1]
    return s


def cluster_data_by_pos(feat_dicts):
    pos_to_indices = {}
    pos_key = 'pos'
    for i, d in enumerate(feat_dicts):
        if pos_key in d:
            s = pos_key + '=' + d[pos_key]
        else:
            s = pos_key + '=' + NULL
        if s in pos_to_indices:
            pos_to_indices[s].append(i)
        else:
            pos_to_indices[s] = [i]
    return pos_to_indices


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_feature_alphabet(feat_dicts):
    feature_alphabet = []
    for f_dict in feat_dicts:
        for f in f_dict:
            feature_alphabet.append(f + ':' + f_dict[f])
    feature_alphabet = list(set(feature_alphabet))
    return feature_alphabet


def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs += align_symbol * (len(ins) - len(outs))
        elif len(outs) > len(ins):
            ins += align_symbol * (len(outs) - len(ins))
            alignedpairs.append((ins, outs))
    return alignedpairs


def mcmc_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol)
    return a.alignedpairs


def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol, mode='med')
    return a.alignedpairs