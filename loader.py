# tools to load data

import numpy as np
import pandas as pd
import re

field_names = [
        'sentence_ix',
        'word',
        'stem',
        'pos',
        '_pos_',
        'meaning',
        'parse_ix',
        'parse_role',
        'blank0',
        'blank1']

lang_map = {
        'en': 'English',
        'fi': 'Finnish',
        'fr': 'French'}

number_re = re.compile('\d')
website_re = re.compile('^http')

def clean(word):
    if pd.isnull(word):
        word = 'NAN'
    elif number_re.search(word):
        word = 'NUMBER'
    elif website_re.match(word):
        word = 'WEBSITE'
    return word

def ud_load(lang_str):
    ud_format = 'ud-treebanks-v2.0/UD_{}/{}-ud-{}.conllu'
    lang_train = pd.read_csv(
            ud_format.format(lang_map[lang_str], lang_str, 'train'),
            names=field_names,
            delimiter='\t')
    lang_dev = pd.read_csv(
            ud_format.format(lang_map[lang_str], lang_str, 'dev'),
            names=field_names,
            delimiter='\t')
    lang = pd.concat(
            [lang_train, lang_dev],
            ignore_index=True)
    if not np.issubdtype(lang.sentence_ix.dtype, np.number):
        lang = lang[~lang.sentence_ix.str.startswith('#')]
        lang = lang[~lang.sentence_ix.str.contains('-')]
        lang['sentence_ix'] = pd.to_numeric(lang['sentence_ix'])
        lang = lang.reset_index(drop=True)
    lang['sent_mark'] = lang.sentence_ix.diff()
    lang_wd = [[]]
    lang_pos = [[]]
    counter = 0
    for row_ix in range(lang.shape[0]):
        row = lang.iloc[row_ix]
        if (not np.isnan(row.sent_mark)) and (row.sent_mark < 0):
            lang_wd.append([])
            lang_pos.append([])
            counter = counter + 1
        word = clean(row.word)
        pos = row.pos
        lang_wd[counter].append(word)
        lang_pos[counter].append(pos)
    return lang_wd, lang_pos

def ep_load(src_str, tgt_str):
    ep_format = 'europarl/tok.lower.{}-{}.{}'
    src_file = open(ep_format.format(tgt_str, src_str, src_str), 'r')
    src_data = []
    for line in src_file.readlines():
        sent = line.strip()
        words = []
        for word in sent.split(' '):
            word = clean(word.strip())
            words.append(word)
        src_data.append(words)
    src_file.close()
    tgt_file = open(ep_format.format(tgt_str, src_str, tgt_str), 'r')
    tgt_data = []
    for line in tgt_file.readlines():
        sent = line.strip()
        words = []
        for word in sent.split(' '):
            word = clean(word.strip())
            words.append(word)
        tgt_data.append(words)
    tgt_file.close()
    return src_data, tgt_data

