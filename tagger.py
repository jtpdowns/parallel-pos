# parallel part-of-speech tagging tools
# (source_word, source_pos,
# source_parallel, target_parallel,
# target_word) => target_pos

import numpy as np

from collections import Counter
from embedding import *
from sklearn.cluster import DBSCAN

NOUN = 'NOUN'

def noun_tagger(
        src_wd,
        src_pos,
        src_para,
        tgt_para,
        tgt_wd):
    tgt_pos = []
    index = 0
    for sent in tgt_wd:
        tgt_pos.append([])
        for wd in sent:
            tgt_pos[index].append(NOUN)
        index = index + 1
    return tgt_pos

def cluster_tagger(
        src_wd,
        src_pos,
        src_para,
        tgt_para,
        tgt_wd,
        embedder=bi_lsa):
    tgt_pos = []
    src_v, tgt_v, embedding = embedder(src_wd, tgt_wd)
    # cluster word vectors
    dbscan = DBSCAN(
            algorithm='brute',
            eps=2e-1,
            metric='cosine')
    clusters = dbscan.fit_predict(embedding)
    # count POS for each cluster
    pos_dict = dict()
    for cluster in np.unique(clusters):
        pos_dict[cluster] = Counter()
    for sent_ix, sent in enumerate(src_wd):
        for word_ix, word in enumerate(sent):
            word_pos = src_pos[sent_ix][word_ix]
            cluster_ix = get_index(src_v, word)
            if cluster_ix != -1:
                cluster = clusters[cluster_ix]
                pos_dict[cluster][word_pos] += 1
    # collect most common POS for each cluster
    max_pos_dict = dict()
    for cluster in np.unique(clusters):
        max_pos_list = pos_dict[cluster].most_common(1)
        if max_pos_list:
            max_pos_dict[cluster] = max_pos_list[0][0]
        else:
            max_pos_dict[cluster] = NOUN
    # tag target words with most common POS
    tgt_pos_dict = dict()
    for word_ix, word in enumerate(tgt_v):
        vector_ix = len(src_v) + word_ix
        cluster_ix = cluster_ix = get_index(src_v, word)
        if cluster_ix != -1:
            cluster = clusters[cluster_ix]
            tgt_pos_dict[word] = max_pos_dict[cluster]
        else:
            tgt_pos_dict[word] = NOUN
    # tag text for validation
    for sent_ix, sent in enumerate(tgt_wd):
        tgt_pos.append([])
        for word in sent:
            if word in tgt_pos_dict:
                word_pos = tgt_pos_dict[word]
            else:
                word_pos = NOUN
            tgt_pos[sent_ix].append(word_pos)
    return tgt_pos

def zsb_rnn_tagger(
        src_wd,
        src_pos,
        src_para,
        tgt_para,
        tgt_wd,
        embedder=bi_lsa):
    tgt_pos = []
    src_v, tgt_v, embedding = embedder(src_wd, tgt_wd)
    return tgt_pos

