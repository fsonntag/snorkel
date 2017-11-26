import torch
import scipy
import numpy as np
from .utils import *


class FtrRepr(object):
    def __init__(self):
        pass


class EmbRepr(FtrRepr):

    def __init__(self, vocab, word_embs, decay=1.0, window=0,
                 n_components=0, alignment=None, cache=False):
        # setup embeddings
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        self.word_embs = np.array(word_embs)
        self.word_emb_dim = self.word_embs.shape[1]
        # start or stop of sequence
        self.terminal = np.zeros(self.word_emb_dim)

        # context weighting
        self.decay = decay
        self.window = window

        # PCA
        self.n_components = n_components
        self.align_func = None if not alignment else self._init_alignment(alignment, n_components, self.word_emb_dim)

        # TODO: implement feature ca
        self.cache = cache

    def _init_alignment(self, method, n_components, dim, rm_top_n=0):

        if method == "magic1":
            magic_theta_1 = np.ones((rm_top_n + n_components) * dim)
            return lambda z: magic_theta(z, magic_theta_1)
        elif method == "magicg":
            magic_theta_g = np.random.normal(0, 1.0, (rm_top_n + n_components) * dim)
            return lambda z: magic_theta(z, magic_theta_g)
        elif method == "procrustes":
            return lambda z: procrustes(z, init_f(rm_top_n, n_components, dim))
        return None

    def prepare_sequence(self, seq):
        return np.array([self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['<OOV>'] for w in seq])

    def get_features(self, seq):

        idxs = self.prepare_sequence(seq)

        if self.window == 0:
            return self.word_embs[idxs]

        ftrs = []
        for k in range(len(seq)):
            # exponential weighting by distance from word k
            w = dist_exp_decay(len(seq), self.decay, k, k + 1)
            w_embs = np.multiply(self.word_embs[idxs], w.reshape(-1, 1))

            # left/right context embs
            i, j = max(k - self.window, 0), min(k + self.window, len(seq))
            l_win = w_embs[i:k]
            r_win = w_embs[k + 1:j + 1]

            # mean features
            if l_win.size == 0:
                l_win    = self.terminal
                l_win_mu = self.terminal
            else:
                l_win_mu = np.mean(l_win, axis=0)
                l_win_mu = l_win_mu / np.linalg.norm(l_win_mu, axis=0, ord=2)

            if r_win.size == 0:
                r_win    = self.terminal
                r_win_mu = self.terminal
            else:
                r_win_mu = np.mean(r_win, axis=0)
                r_win_mu = r_win_mu / np.linalg.norm(r_win_mu, axis=0, ord=2)

            # add top n principal components
            if self.n_components > 0:
                l_win_pc = get_principal_components(l_win - l_win_mu, r=self.n_components, align_func=self.align_func)
                r_win_pc = get_principal_components(r_win - r_win_mu, r=self.n_components, align_func=self.align_func)
                v = np.concatenate((l_win_mu, l_win_pc.ravel(), w_embs[k], r_win_mu, r_win_pc.ravel()))
            else:
                v = np.concatenate((l_win_mu, w_embs[k], r_win_mu))
            ftrs.append(v)
        return ftrs