# set of bilingual word embeddings

from bisect import bisect_left
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import svds

def get_index(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def get_vocabulary(in_wd, n_vocab=None):
    vocabulary = Counter()
    for sent in in_wd:
        for word in sent:
            vocabulary[word] += 1
    most_common = vocabulary.most_common(n_vocab)
    return sorted(pair[0] for pair in most_common)

# super simple
def bi_lsa(src_wd, tgt_wd, n_vocab=10000):
    src_v = get_vocabulary(src_wd, n_vocab=n_vocab)
    tgt_v = get_vocabulary(tgt_wd, n_vocab=n_vocab)
    n_src = len(src_v)
    n_tgt = len(tgt_v)
    n_words = n_src + n_tgt
    n_sents = len(src_wd)
    lsa_store = dok_matrix((n_words, n_sents))
    for sent_ix in range(n_sents):
        for word in src_wd[sent_ix]:
            wd_ix = get_index(src_v, word)
            if wd_ix != -1:
                lsa_store[wd_ix, sent_ix] += 1
        for word in tgt_wd[sent_ix]:
            wd_ix = get_index(tgt_v, word)
            if wd_ix != -1:
                lsa_store[n_src + wd_ix, sent_ix] += 1
    u, s, vt = svds(lsa_store, k=100, return_singular_vectors='u')
    embedding = u
    # keep two vocabularies separate
    # because of the potential for overlap
    return src_v, tgt_v, embedding

# requires word alignment
def bi_skip(src_wd, tgt_wd):
    return []

# enforces sentence-level agreement
def bi_cvm(src_wd, tgt_wd):
    return []

# requires translational lexicon
def bi_cca(src_wd, tgt_wd):
    return []

# requires comparable documents
def bi_vcd(src_wd, tgt_wd):
    return []

