# parallel part-of-speech tagging tools
# (source_word, source_pos,
# source_parallel, target_parallel,
# target_word) => target_pos

from embedding import *

def noun_tagger(
        src_wd,
        src_pos,
        src_para,
        tgt_para,
        tgt_wd):
    tgt_pos = []
    for sent in tgt_wd:
        tgt_pos.append([])
        index = 0
        for wd in sent:
            tgt_pos[index].append('NOUN')
            index = index + 1
    return tgt_pos

def embedding_tagger(
        src_wd,
        src_pos,
        src_para,
        tgt_para,
        tgt_wd,
        embedding=bi_skip):
    tgt_pos = []
    return tgt_pos

