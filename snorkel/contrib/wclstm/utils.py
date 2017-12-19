import numpy as np

import torch
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


def change_marginals_with_spanset_information(candidates, marginals):
    types = candidates[0].values[:-1]

    candidates = [(i, candidate) for i, candidate in enumerate(candidates)]
    candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start, c[1][0].char_end))

    for type in types:
        type_label = candidates[0][1].values.index(type)
        current_spanset = []
        for i, (original_i, candidate) in enumerate(candidates):
            if np.argmax(marginals[original_i]) == type_label:
                if current_spanset == []:
                    current_spanset.append((original_i, candidate))
                else:
                    last_candidate = current_spanset[-1][1]
                    if last_candidate[0].char_end > candidate[0].char_start:
                        current_spanset.append((original_i, candidate))
                    else:
                        change_lower_spanset_probabilities(current_spanset, marginals, type_label)
                        current_spanset = [(original_i, candidate)]
        change_lower_spanset_probabilities(current_spanset, marginals, type_label)


def change_lower_spanset_probabilities(current_spanset, marginals, type_label):
    if not current_spanset or len(current_spanset) == 1:
        return
    marginal_indices = np.asarray([css[0] for css in current_spanset])
    max_index = marginal_indices[np.argmax(marginals[marginal_indices, type_label])]
    non_max_indices = marginal_indices[np.argwhere(marginal_indices != max_index)]
    marginals[non_max_indices, -1] = 1.
