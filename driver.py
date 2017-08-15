# run part-of-speech tagging for languages
# and print scores

from loader import *
from tagger import *
from score import *

if __name__ == '__main__':
    # load ud data
    ud_en_wd, ud_en_pos = ud_load('en')
    ud_fi_wd, fi_true_pos = ud_load('fi')
    ud_fr_wd, fr_true_pos = ud_load('fr')
    # load ep data
    ep_fien_en_wd, ep_fien_fi_wd = ep_load('en', 'fi')
    ep_fren_en_wd, ep_fren_fr_wd = ep_load('en', 'fr')
    # run tagger
    tagger = noun_tagger
    fi_pred_pos = tagger(
            ud_en_wd,
            ud_en_pos,
            ep_fien_fi_wd,
            ep_fien_en_wd,
            ud_fi_wd)
    fr_pred_pos = tagger(
            ud_en_wd,
            ud_en_pos,
            ep_fren_fr_wd,
            ep_fren_en_wd,
            ud_fr_wd)
    # score tags
    fi_score = score(fi_pred_pos, fi_true_pos)
    fr_score = score(fr_pred_pos, fr_true_pos)
    print('Finnish score: {}'.format(fi_score))
    print('French score: {}'.format(fr_score))

