import numpy as np
import torch
from torch.autograd import Variable

from snorkel.utils import overlapping_score


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
    return tokens[max(candidate_start - candidate_radius, 0):
                  min(candidate_end + 1 + candidate_radius + 2, len(tokens))]


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


def pad_batch(batch_w_context, batch_w_candidate, batch_c_candidate, context_radius, max_word_length):
    """Pad the batch into matrix"""
    batch_size = len(batch_w_context)
    max_cand_len = 8
    max_word_len = min(int(np.max([len(w) for words in batch_c_candidate for w in words])), max_word_length)

    context_word_matrix = np.zeros((batch_size, 2 * context_radius + 3), dtype=np.int)
    candidate_word_matrix = np.zeros((batch_size, max_cand_len), dtype=np.int)
    candidate_char_matrix = np.zeros((batch_size, max_cand_len, max_word_len), dtype=np.int)

    for idx1, i in enumerate(batch_w_context):
        for idx2, j in enumerate(i):
            try:
                context_word_matrix[idx1, idx2] = j
            except IndexError:
                pass
    context_word_matrix = Variable(torch.from_numpy(context_word_matrix))
    context_word_mask_matrix = Variable(torch.eq(context_word_matrix.data, 0))

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

    return context_word_matrix, context_word_mask_matrix, \
           candidate_word_matrix, candidate_word_mask_matrix, \
           candidate_char_matrix, candidate_char_mask_matrix


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
                    if last_candidate[0].sentence_id == candidate[0].sentence_id \
                            and last_candidate[0].char_end > candidate[0].char_start:
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


def merge_to_spansets(candidates, marginals):
    candidate_spansets = []
    marginal_spansets = []
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
                        current_marginals = marginals_for_spanset(current_spanset, marginals)
                        marginal_spansets.append(current_marginals)
                        candidate_spansets.append(current_spanset)
                        current_spanset = [(original_i, candidate)]
        current_marginals = marginals_for_spanset(current_spanset, marginals)
        marginal_spansets.append(current_marginals)
        candidate_spansets.append(current_spanset)
    return candidate_spansets, marginal_spansets


def merge_to_spansets_train(X, train_marginals, pred_marginals):
    candidate_spansets = ([], [])
    Y_pred = ([], [])

    for pred_i, marginals in enumerate([train_marginals, pred_marginals]):
        candidates = [(i, candidate) for i, candidate in enumerate(X)]
        candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start, c[1][0].char_end))
        current_spanset = []
        cardinality = marginals.shape[1]
        for value in range(cardinality - 1):
            for i, (original_i, candidate) in enumerate(candidates):
                if marginals[original_i].argmax() == value:
                    if current_spanset == []:
                        current_spanset.append((original_i, candidate))
                    else:
                        last_candidate = current_spanset[-1][1]
                        if overlapping_score(last_candidate, candidate) > 0:
                            current_spanset.append((original_i, candidate))
                        else:
                            spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
                            for spanset_chunk in spanset_chunks:
                                pred_y = y_from_spanset_chunk(spanset_chunk, marginals, value)
                                candidate_spansets[pred_i].append(spanset_chunk)
                                Y_pred[pred_i].append(pred_y)
                            current_spanset = [(original_i, candidate)]

            spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
            for spanset_chunk in spanset_chunks:
                pred_y = y_from_spanset_chunk(spanset_chunk, marginals, value)
                candidate_spansets[pred_i].append(spanset_chunk)
                Y_pred[pred_i].append(pred_y)
            current_spanset = []

        for i, (original_i, candidate) in enumerate(candidates):
            if marginals[original_i].argmax() + 1 == cardinality:
                candidate_spansets[pred_i].append([(original_i, candidate)])
                Y_pred[pred_i].append(np.zeros(1))

    spansets_true, spansets_pred = candidate_spansets
    Y_true, Y_pred = Y_pred
    assert len(spansets_true) == len(Y_true)
    assert len(spansets_pred) == len(Y_pred)
    return spansets_true, Y_true, spansets_pred, Y_pred


def y_from_spanset_chunk(spanset_chunk, marginals, value):
    spanset_marginals = np.asarray([marginals[s[0]] for s in spanset_chunk])
    best_candidate_row = np.argmax(spanset_marginals[:, value])
    pred_y = np.zeros(len(spanset_chunk))
    pred_y[best_candidate_row] = value + 1

    return pred_y


def merge_to_spansets_dev(X, Y, marginals):
    candidate_spansets = []
    Y_true = []
    Y_pred = []

    candidates = [(i, candidate) for i, candidate in enumerate(X)]
    candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start, c[1][0].char_end))
    current_spanset = []
    cardinality = marginals.shape[1]
    for value in range(cardinality - 1):
        for i, (original_i, candidate) in enumerate(candidates):
            if marginals[original_i].argmax() == value:
                if current_spanset == []:
                    current_spanset.append((original_i, candidate))
                else:
                    last_candidate = current_spanset[-1][1]
                    if overlapping_score(last_candidate, candidate) > 0:
                        current_spanset.append((original_i, candidate))
                    else:
                        spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
                        for spanset_chunk in spanset_chunks:
                            true_y, pred_y = ys_from_spanset_chunk(spanset_chunk, Y, marginals, value)
                            candidate_spansets.append(spanset_chunk)
                            Y_true.append(true_y)
                            Y_pred.append(pred_y)
                        current_spanset = [(original_i, candidate)]

        spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
        for spanset_chunk in spanset_chunks:
            true_y, pred_y = ys_from_spanset_chunk(spanset_chunk, Y, marginals, value)
            candidate_spansets.append(spanset_chunk)
            Y_true.append(true_y)
            Y_pred.append(pred_y)
        current_spanset = []

    for i, (original_i, candidate) in enumerate(candidates):
        if marginals[original_i].argmax() + 1 == cardinality:
            candidate_spansets.append([(original_i, candidate)])
            Y_true.append(np.ravel(Y[original_i].todense()))
            Y_pred.append(np.zeros(1))

    assert len(candidate_spansets) == len(Y_pred)
    assert len(candidate_spansets) == len(Y_true)
    return candidate_spansets, Y_true, Y_pred


def ys_from_spanset_chunk(spanset_chunk, Y, marginals, value):
    true_y = np.ravel([Y[s[0]].todense() for s in spanset_chunk])

    spanset_marginals = np.asarray([marginals[s[0]] for s in spanset_chunk])
    best_candidate_row = np.argmax(spanset_marginals[:, value])
    pred_y = np.zeros(len(spanset_chunk))
    pred_y[best_candidate_row] = value + 1

    return true_y, pred_y


def marginals_for_spanset(current_spanset, marginals):
    if not current_spanset:
        return
    marginal_indices = [css[0] for css in current_spanset]
    return marginals[marginal_indices]
