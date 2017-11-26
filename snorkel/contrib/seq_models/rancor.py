import torch
import re
import sys
import nltk
import collections
import numpy as np
from .embs import Embeddings
import itertools
import scipy.linalg
import numpy as np


def match_term(t, embs):
    """
    Given a string t, find best match in dictionary embs
    :param t:
    :param embs:
    :return:
    """
    if t in embs:
        return embs[t]
    if t.lower() in embs:
        return embs[t.lower()]

    # strip punctuation, e.g., (ABC) --> ABC
    rgx = re.compile('(^[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])|([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]$)')
    nt = re.sub(rgx,"",t)
    if nt in embs:
        return embs[nt]
    return None


def get_singular_vectors(x, r, l=0, method=None):
    """
    :param x: embedding matrix
    :param r: keep top r components
    :param l: remove top l components
    :param method:
    :return:
    """
    u, s, v = torch.svd(x)
    k = r if v.size(1) > l + r else v.size(1) - l
    z = torch.zeros(l + r, v.size(0)).double()
    z[0:l + k, ] = v.transpose(0, 1)[:l + k, ]

    if method in ['magic1', 'magicg']:
        theta = np.random.normal(0, 1.0, (l + r) * x.shape[1])
        theta = torch.from_numpy(theta[:z.size(0) * z.size(1)]).view(z.size(0), z.size(1)).double()
        z = torch.mul(z, torch.sign(torch.mul(z, theta)))

    # TODO: Add this back in
    # if method == 'procrustes':
    #     r = \
    #     scipy.linalg.orthogonal_procrustes(z.numpy().T, self.F[:z.size(0) * z.size(1)].reshape(z.size(0), z.size(1)).T)[
    #         0]
    #     z = torch.mm(torch.from_numpy(r), z)

    return z[l:, ]


def get_principal_components(X, k=1, method=None):
    """
    Compute k principal components of X
    :param X:
    :param k:
    :param method:
    :return:
    """
    W = torch.from_numpy(X).float()
    W = W / torch.norm(W)
    mu = torch.mean(W, 1, keepdim=True)
    pc = get_singular_vectors(W - mu, k, method=method).view(1, -1)
    pc = pc / torch.norm(pc)
    return pc.numpy()


def all_but_the_top(W):
    """
    All-but-the-Top transformation
    :param W:
    :return:
    """
    norm = np.linalg.norm(W, axis=1, ord=2)
    W = (W.T / norm).T
    mu = np.mean(W, axis=0)

    W_hat = W - mu
    D = max(1, int(W.shape[1] / 100.0))
    u = get_principal_components(W_hat, D).reshape(D,-1)

    for i in range(W.shape[0]):
        w = np.zeros((1, W.shape[1]),dtype=np.float64)
        for j in range(D):
             w += u[j].T.reshape(-1,1).dot(W[i].reshape(1,-1)).dot(u[j].reshape(-1,1)).T
        W[i] = W_hat[i] - w

    norm = np.linalg.norm(W, axis=1, ord=2)
    W = (W.T / norm).T

    return W


def embed_terms(terms, embs, mode="pca"):
    """
    Create embeddings for provided term list.
    Multi-word terms are constructed using a using neural bag-of-words assumption, i.e.,
     given a stacked word embedding matrix w
     1) mean(w)
     2) pca(w) -> k1 ... kn  concatenate top k principle components

    :param terms:
    :param embs:
    :return:
    """
    dim = len(embs[embs.keys()[0]])
    W = np.zeros((len(terms), dim), dtype=np.float64)
    oov = np.random.rand(1,dim).astype(np.float64)

    for i, term in enumerate(terms):
        w = []
        for t in term.split():
            v = match_term(t, embs)
            if type(v) != np.ndarray:
                continue
            w.append(v)
        if not w:
            w.append(oov)

        if mode != "mean":
            W[i] = get_principal_components(np.array(w), k=1, method=None) if len(w) > 1 else np.array(w)
        else:
            W[i] = np.mean(np.array(w), axis=0)

    norm = np.linalg.norm(W, axis=1, ord=2)
    return (W.T / norm).T


def get_sentence_vocab(sentences, max_kgrams=4):
    terms = collections.defaultdict(dict)
    for s in sentences:
        for k in range(1, max_kgrams+1):
            tokens = list(nltk.ngrams(s.words,k))
            tokens = dict.fromkeys([" ".join(w) for w in tokens])
            terms[k].update(tokens)
    return terms


class EntityRancor(object):

    def __init__(self, embeddings, vocab=None, all_but_the_top=True, verbose=True):
        """

        :param embeddings:
        :param vocab:
        :param all_but_the_top:
        :param verbose:
        """
        self.verbose = verbose
        self.all_but_the_top = all_but_the_top

        if type(embeddings) is str:
            fmt = self._get_emb_format(embeddings.split(".")[-1])
            self.embs = Embeddings(embeddings, fmt=fmt)

        elif type(embeddings) is Embeddings:
            self.embs = embeddings #self._load_embs(embeddings)

        # if a dictionary is provided, sort by token length and preload embs
        if vocab:
            self.vocab = vocab
            # TODO
            raise NotImplementedError()


    def _get_emb_format(self, ext):

        if ext == "bin":
            return "gensim"
        elif ext == "vec":
            return "fasttext"
        else:
            return "text"


    def _expand_vocab(self, vocab):
        """
        Add additional terms based on stripping left/right punctuation
        (ABC) --> ABC
        :return:
        """
        terms = []
        rgx = re.compile('(^[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])|([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]$)')
        for t in vocab:
            nt = re.sub(rgx, "", t)
            if nt not in vocab:
                terms.append(nt)
        vocab.update(dict.fromkeys(terms))
        return vocab


    def _load_embs(self, vocab, embs):
        """
        Only load embeddings that are found in our vocab dictionary
        normalized by lettercase
        :param embs:
        :return:
        """
        s_embs = {}
        for term,vec in embs:
            if term in vocab or term.lower() in vocab:
                s_embs[term] = vec
        return s_embs


    def _get_vocab(self, sentences, max_ngrams=4):
        """

        :param sentences:
        :param max_ngrams:
        :return:
        """
        terms = collections.defaultdict(dict)
        for s in sentences:
            for k in range(1, max_ngrams + 1):
                tokens = list(nltk.ngrams(s.words, k))
                tokens = [" ".join(w) for w in tokens]
                tokens = dict.fromkeys([t.strip() for t in tokens if t.strip() != ""])
                terms[k].update(tokens)
        return terms


    def query(self, queries, sentences, top_k=10, max_ngrams=4, mode="mean"):
        """
        This queries unique dictionary kgrams (*not* specific candidate instances)

        :param queries:
        :param sentences:
        :param top_k:
        :param max_ngrams:
        :param mode:
        :return:
        """
        # build vocabulary dictionary
        vocab    = self._get_vocab(sentences, max_ngrams=max_ngrams)
        # add query terms
        for q in queries:
            for k in range(1, max_ngrams + 1):
                tokens = list(nltk.ngrams(q.split(), k))
                tokens = [" ".join(w) for w in tokens]
                tokens = dict.fromkeys([t.strip() for t in tokens if t.strip() != ""])
                vocab[k].update(tokens)

        vocab[1] = self._expand_vocab(vocab[1])
        # filter embeddings to dictionary terms
        s_embs = self._load_embs(vocab[1], self.embs)

        if self.verbose:
            print("=" * 40)
            print("Ranking")
            print("=" * 40)
            print("top_k:      {:>8}".format(top_k))
            print("max_ngrams: {:>8}".format(max_ngrams))
            print("mode:       {:>8}".format(mode))
            print("|Vocab|:    {:>8}".format(len(vocab[1])))
            print("|Embs|:     {:>8}".format(len(s_embs)))
            if self.all_but_the_top:
                print(" -- Using 'All-but-the-Top' transform, see (Mu et al. 2017)")
            print("|Queries|:  {:>8}".format(len(queries)))
            print("-" * 40)

        if self.all_but_the_top:
            embs = np.array(s_embs.values())
            embs = all_but_the_top(embs)
            s_embs = {term:embs[i] for i,term in enumerate(s_embs.keys())}

        scores = collections.defaultdict(list)

        for ngram in sorted(vocab):
            V = np.array(vocab[ngram].keys())
            W = embed_terms(vocab[ngram], s_embs, mode=mode)
            q = embed_terms(queries.keys(), s_embs, mode=mode)
            r = q.dot(W.T)

            # top k query matches
            for i,query in enumerate(queries.keys()):
                idxs = np.argsort(-r[i])[:top_k]
                scores[query].extend(zip(r[i][idxs], V[idxs]))

            if self.verbose:
                total = set(itertools.chain.from_iterable([zip(*scores[q])[-1] for q in scores]))
                print(" -- {}grams complete... (matches={})".format(ngram,len(total)))

        # merge results and rank by cosine angle
        spans = []
        for q in scores:
            spans.extend(zip(*sorted(scores[q], reverse=1)[:top_k])[-1])
        spans = dict.fromkeys(spans)
        print("{} queries returned {} matches".format(len(queries), len(spans)))

        # extract actual candidate spans
        candidates = []
        for ngram in sorted(vocab):
            for i,s in enumerate(sentences):
                # respect original char offsets
                terms = []
                for tokens in nltk.ngrams(zip(s.words, s.char_offsets), ngram):
                    txt, start = tokens[0]
                    for t, offset in tokens[1:]:
                        txt += " " * (offset - start - len(txt)) + t if len(txt) != offset else t
                    terms.append(txt)

                abs_char_offsets = list(nltk.ngrams(s.abs_char_offsets, ngram))

                for j,t in enumerate(terms):
                    if t in spans or t.lower() in spans:
                        t = t.lower() if t not in spans else t
                        candidates.append((s.document, s.position, abs_char_offsets[j][0], t))

        return spans, candidates

