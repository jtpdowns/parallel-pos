# parallel part-of-speech tagging tools
# (source_word, source_pos,
# source_parallel, target_parallel,
# target_word) => target_pos

import numpy as np

from embedding import *
from sklearn.cluster import DBSCAN

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
            tgt_pos[index].append('NOUN')
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
    dbscan = DBSCAN()
    cluster_labels = dbscan.fit_predict(embedding)
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

