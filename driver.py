# run part-of-speech tagging for languages
# and print scores

from loader import *
from tagger import *
from score import *

if __name__ == '__main__':
    # load ud data
    print('Loading tagged data...')
    ud_en_wd, ud_en_pos = ud_load('en')
    ud_fi_wd, fi_true_pos = ud_load('fi')
    print('Tagged data loaded...')
    # load ep data
    print('Loading parallel data...')
    ep_fien_en_wd, ep_fien_fi_wd = ep_load('en', 'fi')
    print('Parallel data loaded...')
    # run taggers
    for tagger in [noun_tagger, cluster_tagger]:
        print('Running tagger...')
        fi_pred_pos = tagger(
                ud_en_wd,
                ud_en_pos,
                ep_fien_fi_wd,
                ep_fien_en_wd,
                ud_fi_wd)
        print('Tagger complete...')
        # score tags
        print('Scoring tags...')
        fi_score = score(fi_pred_pos, fi_true_pos)
        print('{} score on Finnish: {}'.format(
            tagger.__name__,
            fi_score))

