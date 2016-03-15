NULL = '%'

def cluster_data_by_morph_type(feat_dicts, feats):
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