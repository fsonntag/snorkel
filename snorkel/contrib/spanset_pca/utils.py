import numpy as np
import torch
import scipy.sparse as sparse


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

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


def merge_to_spansets(candidates, marginals):
    candidate_spansets = []
    marginal_spansets = []
    marginal_picks = []
    marginal_picks_ext = []

    candidates = [(i, candidate) for i, candidate in enumerate(candidates)]
    candidates.sort(key=lambda c: (c[1][-1][0].sentence_id, c[1][-1][0].char_start, c[1][-1][0].char_end))
    current_spanset = []
    for i, (original_i, candidate) in enumerate(candidates):
        if current_spanset == []:
            current_spanset.append((original_i, candidate))
        else:
            last_candidate = current_spanset[-1][1]
            if last_candidate[-1][0].sentence_id == candidate[-1][0].sentence_id \
                and last_candidate[-1][0].char_end > candidate[-1][0].char_start:
                current_spanset.append((original_i, candidate))
            else:
                current_marginals = marginals_for_spanset(current_spanset, marginals)
                marginal_spansets.append(current_marginals)
                candidate_spansets.append(current_spanset)
                current_spanset = [(original_i, candidate)]

    current_marginals = marginals_for_spanset(current_spanset, marginals)
    marginal_spansets.append(current_marginals)
    candidate_spansets.append(current_spanset)
    max_spanset_size = max([len(candidate_spanset) for candidate_spanset in candidate_spansets])
    for marginals, candidate_spanset in zip(marginal_spansets, candidate_spansets):
        current_picks = picks_from_marginals(marginals, max_spanset_size)
        marginal_picks.append(current_picks)
        current_picks_ext = sparse.lil_matrix((1, len(candidate_spanset)), dtype=np.int)
        for i in range(current_picks.shape[0]):
            if current_picks[i] < max_spanset_size:
                current_picks_ext[0, current_picks[i]] = i + 1
        marginal_picks_ext.append(current_picks_ext)

    return candidate_spansets, marginal_spansets, marginal_picks, marginal_picks_ext

def picks_from_marginals(marginals, max_spanset_size):
    picks = np.zeros(marginals.shape[1] - 1, dtype=int)
    for i in range(marginals.shape[1] - 1):
        max_value, max_row = torch.max(marginals[:, i], dim=0)
        if (torch.max(marginals[max_row]) == max_value)[0]:
            picks[i] = max_row
        else:
            picks[i] = max_spanset_size
    return picks

def merge_to_spansets_dev(X_dev, Y_dev):
    candidate_spansets = []
    spanset_y = []

    candidates = [(i, candidate) for i, candidate in enumerate(X_dev)]
    candidates.sort(key=lambda c: (c[1][-1][0].sentence_id, c[1][-1][0].char_start, c[1][-1][0].char_end))
    current_spanset = []
    for i, (original_i, candidate) in enumerate(candidates):
        if current_spanset == []:
            current_spanset.append((original_i, candidate))
        else:
            last_candidate = current_spanset[-1][1]
            if last_candidate[-1][0].sentence_id == candidate[-1][0].sentence_id \
                    and last_candidate[-1][0].char_end > candidate[-1][0].char_start:
                current_spanset.append((original_i, candidate))
            else:
                current_y = sparse.lil_matrix(Y_dev[[s[0] for s in current_spanset]]).T
                spanset_y.append(current_y)
                candidate_spansets.append(current_spanset)
                current_spanset = [(original_i, candidate)]
    current_y = sparse.lil_matrix(Y_dev[[s[0] for s in current_spanset]]).T
    spanset_y.append(current_y)
    candidate_spansets.append(current_spanset)
    return candidate_spansets, spanset_y



def merge_to_spansets_by_type(candidates, marginals):
    candidate_spansets = []
    marginal_spansets = []
    marginal_picks = []
    types = candidates[0][-1].values[:-1]

    candidates = [(i, candidate) for i, candidate in enumerate(candidates)]
    candidates.sort(key=lambda c: (c[1][-1][0].sentence_id, c[1][-1][0].char_start, c[1][-1][0].char_end))
    # TODO no negative samples
    for type in types:
        type_label = candidates[0][1][-1].values.index(type)
        current_spanset = []
        for i, (original_i, candidate) in enumerate(candidates):
            if np.argmax(marginals[original_i]) == type_label:
                if current_spanset == []:
                    current_spanset.append((original_i, candidate))
                else:
                    last_candidate = current_spanset[-1][1]
                    if last_candidate[-1][0].sentence_id == candidate[-1][0].sentence_id \
                        and last_candidate[-1][0].char_end > candidate[-1][0].char_start:
                        current_spanset.append((original_i, candidate))
                    else:
                        current_marginals = marginals_for_spanset(current_spanset, marginals)
                        if current_marginals.shape[0] == 1:
                            marginal_pick = 0
                        else:
                            marginal_pick = current_marginals[:, type_label].max(dim=0)[1][0]
                        marginal_picks.append(marginal_pick)
                        marginal_spansets.append(current_marginals)
                        candidate_spansets.append(current_spanset)
                        current_spanset = [(original_i, candidate)]
        current_marginals = marginals_for_spanset(current_spanset, marginals)
        if current_marginals.shape[0] == 1:
            marginal_pick = 0
        else:
            marginal_pick = current_marginals[:, type_label].max(dim=0)[1][0]
        marginal_picks.append(marginal_pick)
        marginal_spansets.append(current_marginals)
        candidate_spansets.append(current_spanset)
    return candidate_spansets, marginal_spansets, marginal_picks

def marginals_for_spanset(current_spanset, marginals):
    if not current_spanset:
        return
    marginal_indices = [css[0] for css in current_spanset]
    return marginals[marginal_indices]