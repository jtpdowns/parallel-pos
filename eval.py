# tool for evaluation of part-of-speech tags
# based on per-token accuracy

def score(tag_pos, tgt_pos):
    correct = 0
    count = 0
    for sent_ix in range(len(tgt_pos)):
        pred_sent = tag_pos[sent_ix]
        true_sent = tgt_pos[sent_ix]
        for word_ix in range(len(true_sent)):
            pred_pos = pred_sent[word_ix]
            true_pos = true_sent[word_ix]
            if pred_pos == true_pos:
                correct = correct + 1
            count = count + 1
    return correct/count

