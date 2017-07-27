import pandas as pd

### LOAD DATASET ###
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

tag_set = {
        'ADJ',
        'ADP',
        'ADV',
        'AUX',
        'CCONJ',
        'DET',
        'INTJ',
        'NOUN',
        'NUM',
        'PART',
        'PRON',
        'PROPN',
        'PUNCT',
        'SCONJ',
        'SYM',
        'VERB',
        'X'}

english_train = pd.read_csv(
        'ud-treebanks-v2.0/UD_English/en-ud-train.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')
finnish_train = pd.read_csv(
        'ud-treebanks-v2.0/UD_Finnish/fi-ud-train.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')
french_train = pd.read_csv(
        'ud-treebanks-v2.0/UD_French/fr-ud-train.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')

english_dev = pd.read_csv(
        'ud-treebanks-v2.0/UD_English/en-ud-dev.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')
finnish_dev = pd.read_csv(
        'ud-treebanks-v2.0/UD_Finnish/fi-ud-dev.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')
french_dev = pd.read_csv(
        'ud-treebanks-v2.0/UD_French/fr-ud-dev.conllu',
        names=field_names,
        delimiter='\t',
        comment='#')

# no training data for the "low-resource" languages
finnish = pd.concat(
        [finnish_train, finnish_dev],
        ignore_index=True)
french = pd.concat(
        [french_train, french_dev],
        ignore_index=True)

# filter down to tag set
english_train = english_train[english_train.pos.isin(tag_set)]
english_dev = english_dev[english_dev.pos.isin(tag_set)]
finnish = finnish[finnish.pos.isin(tag_set)]
french = french[french.pos.isin(tag_set)]

### EXPLORE BASIC COUNTS ###
print('Noun Tagger')
print('English training accuracy = {}'.format((english_train.pos == 'NOUN').mean()))
print('English dev accuracy = {}'.format((english_dev.pos == 'NOUN').mean()))
print('Finnish accuracy = {}'.format((finnish.pos == 'NOUN').mean()))
print('French accuracy = {}'.format((french.pos == 'NOUN').mean()))
print('')


