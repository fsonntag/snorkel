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


def trim_with_radius(tokens, candidate, candidate_radius):
    candidate_start = candidate[0].get_word_start()
    candidate_end = candidate[0].get_word_end()
    return tokens[max(candidate_start - candidate_radius, 0):candidate_start], \
           tokens[candidate_end + 3: min(candidate_end + 1 + candidate_radius + 2, len(tokens))]


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


def pad_batch(left_batch_w_context, right_batch_w_context, batch_w_candidate, batch_c_candidate, context_radius,
              max_word_length):
    """Pad the batch into matrix"""
    batch_size = len(left_batch_w_context)
    max_cand_len = 8
    max_word_len = min(int(max(len(w) for words in batch_c_candidate for w in words)), max_word_length)
    left_max_context_len = max(len(b) for b in left_batch_w_context)
    right_max_context_len = max(len(b) for b in right_batch_w_context)

    left_context_word_matrix = np.zeros((batch_size, left_max_context_len), dtype=np.int)
    right_context_word_matrix = np.zeros((batch_size, right_max_context_len), dtype=np.int)
    candidate_word_matrix = np.zeros((batch_size, max_cand_len), dtype=np.int)
    candidate_char_matrix = np.zeros((batch_size, max_cand_len, max_word_len), dtype=np.int)

    for idx1, i in enumerate(left_batch_w_context):
        for idx2, j in enumerate(i):
            try:
                left_context_word_matrix[idx1, idx2] = j
            except IndexError:
                pass
    left_context_word_matrix = Variable(torch.from_numpy(left_context_word_matrix))
    left_context_word_mask_matrix = Variable(torch.eq(left_context_word_matrix.data, 0))

    for idx1, i in enumerate(right_batch_w_context):
        for idx2, j in enumerate(i):
            try:
                right_context_word_matrix[idx1, idx2] = j
            except IndexError:
                pass
    right_context_word_matrix = Variable(torch.from_numpy(right_context_word_matrix))
    right_context_word_mask_matrix = Variable(torch.eq(right_context_word_matrix.data, 0))

    for idx1, i in enumerate(batch_w_candidate):
        for idx2, j in enumerate(i):
            try:
                candidate_word_matrix[idx1, idx2] = j
            except IndexError:
                pass
    candidate_word_matrix = Variable(torch.from_numpy(candidate_word_matrix))
    candidate_word_mask_matrix = Variable(torch.eq(candidate_word_matrix.data, 0))

    for idx1, i in enumerate(batch_c_candidate):
        for idx2, j in enumerate(i):
            for idx3, k in enumerate(j):
                try:
                    candidate_char_matrix[idx1, idx2, idx3] = batch_c_candidate[idx1][idx2][idx3]
                except IndexError:
                    pass
    candidate_char_matrix = Variable(torch.from_numpy(candidate_char_matrix))
    candidate_char_mask_matrix = Variable(torch.eq(candidate_char_matrix.data, 0))

    return left_context_word_matrix, left_context_word_mask_matrix, \
           right_context_word_matrix, right_context_word_mask_matrix, \
           candidate_word_matrix, candidate_word_mask_matrix, \
           candidate_char_matrix, candidate_char_mask_matrix


def write_attention(X_candidates, left_context_weights, right_context_weights, candidate_weights, context_radius,
                    output_path, limit=1000):
    context_path = output_path / 'context_attention'
    context_path.mkdir(exist_ok=True)
    candidate_path = output_path / 'candidate_attention'
    candidate_path.mkdir(exist_ok=True)
    fig = plt.figure()
    print('Writing attention figures...')
    for i, (candidate, left_context_weight, right_context_weight, candidate_weight) in enumerate(
            zip(X_candidates, left_context_weights, right_context_weights, candidate_weights)):
        if len(candidate) == 2:
            args = [
                (candidate[0].get_word_start(), candidate[0].get_word_end(), 1),
                (candidate[1].get_word_start(), candidate[1].get_word_end(), 2)
            ]
        else:
            args = [(candidate[0].get_word_start(), candidate[0].get_word_end(), 1)]
        left_s_words, right_s_words = trim_with_radius(mark_sentence(candidate_to_tokens(candidate), args), candidate,
                                                       context_radius)
        if left_s_words:
            context_weight = left_context_weight[:len(left_s_words)]
            ax = fig.add_subplot(111)
            cax = ax.matshow(np.reshape(minmax_scale(context_weight), (1, -1)), cmap='afmhot_r', vmin=0, vmax=1)
            fig.colorbar(cax)
            ax.set_xticklabels([''] + left_s_words, rotation=90)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.get_yaxis().set_visible(False)
            span = candidate[0].get_span()
            span = span.replace('/', '.')
            plt.savefig(str((context_path / f'{i}_left_{span}.png').absolute()))
            fig.clf()

        if right_s_words:
            context_weight = right_context_weight[:len(right_s_words)]
            ax = fig.add_subplot(111)
            cax = ax.matshow(np.reshape(minmax_scale(context_weight), (1, -1)), cmap='afmhot_r', vmin=0, vmax=1)
            fig.colorbar(cax)
            ax.set_xticklabels([''] + right_s_words, rotation=90)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.get_yaxis().set_visible(False)
            span = candidate[0].get_span()
            span = span.replace('/', '.')
            plt.savefig(str((context_path / f'{i}_right_{span}.png').absolute()))
            fig.clf()

        c_words = candidate[0].get_attrib_tokens()
        candidate_weight = candidate_weight[:len(c_words)]
        ax = fig.add_subplot(111)
        cax = ax.matshow(np.reshape(minmax_scale(candidate_weight), (1, -1)), cmap='afmhot_r', vmin=0, vmax=1)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + c_words, rotation=90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.get_yaxis().set_visible(False)
        span = candidate[0].get_span()
        span = span.replace('/', '.')
        plt.savefig(str((candidate_path / f'{i}_{span}.png').absolute()))
        fig.clf()

        if i == limit:
            break