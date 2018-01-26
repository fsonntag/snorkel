import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from matplotlib import ticker
from sklearn.preprocessing import minmax_scale
from torch.autograd import Variable


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""

    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s = starting_symbol
        self.unknown = unknown_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.iteritems()}


def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)


def candidate_to_tokens(candidate, token_type='words', lowercase=False):
    tokens = candidate.get_parent().__dict__[token_type]
    return [scrub(w).lower() if lowercase else scrub(w) for w in tokens]


def mark(l, h, idx):
    """Produce markers based on argument positions

    :param l: sentence position of first word in argument
    :param h: sentence position of last word in argument
    :param idx: argument index (1 or 2)
    """
    return [(l, "{}{}".format('~~[[', idx)), (h + 1, "{}{}".format(idx, ']]~~'))]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence

    :param s: list of tokens in sentence
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments

    Example: Then Barack married Michelle.
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


def pad_batch(batch_w, batch_c, max_sentence_length, max_word_length):
    """Pad the batch into matrix"""
    batch_size = len(batch_w)
    max_sent_len = min(int(np.max([len(x) for x in batch_w])), max_sentence_length)
    max_word_len = min(int(np.max([len(w) for words in batch_c for w in words])), max_word_length)
    sent_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)
    word_matrix = np.zeros((batch_size, max_sent_len, max_word_len), dtype=np.int)
    for idx1, i in enumerate(batch_w):
        for idx2, j in enumerate(i):
            try:
                sent_matrix[idx1, idx2] = j
            except IndexError:
                pass
    sent_matrix = Variable(torch.from_numpy(sent_matrix))
    sent_mask_matrix = Variable(torch.eq(sent_matrix.data, 0))

    for idx1, i in enumerate(batch_c):
        for idx2, j in enumerate(i):
            for idx3, k in enumerate(j):
                try:
                    word_matrix[idx1, idx2, idx3] = batch_c[idx1][idx2][idx3]
                except IndexError:
                    pass
    word_matrix = Variable(torch.from_numpy(word_matrix))
    word_mask_matrix = Variable(torch.eq(word_matrix.data, 0))
    return sent_matrix, sent_mask_matrix, word_matrix, word_mask_matrix


def write_attention(X_candidates, all_word_weights, output_path):
    attention_path = output_path / 'attention'
    attention_path.mkdir(exist_ok=True)
    fig = plt.figure()
    print('Writing attention figures...')
    for i, (candidate, sentence_weight) in enumerate(zip(X_candidates, all_word_weights)):
        if len(candidate) == 2:
            args = [
                (candidate[0].get_word_start(), candidate[0].get_word_end(), 1),
                (candidate[1].get_word_start(), candidate[1].get_word_end(), 2)
            ]
        else:
            args = [(candidate[0].get_word_start(), candidate[0].get_word_end(), 1)]
        s_words = mark_sentence(candidate_to_tokens(candidate), args)
        context_weight = sentence_weight[:len(s_words)]
        ax = fig.add_subplot(111)
        cax = ax.matshow(np.reshape(minmax_scale(context_weight), (1, -1)), cmap='afmhot_r', vmin=0, vmax=1)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + s_words, rotation=90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.get_yaxis().set_visible(False)
        span = candidate[0].get_span()
        span = span.replace('/', '.')
        plt.savefig(str((attention_path / f'{i}_{span}.png').absolute()))
        fig.clf()
