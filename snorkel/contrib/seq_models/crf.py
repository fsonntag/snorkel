#
#
# Based on http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
#
#
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import numpy as np
from .utils import *
import logging

logger = logging.getLogger(__name__)


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_idx, requires_grad=False, host_device="cpu"):
    idxs = [to_idx[w] if w in to_idx else to_idx['<OOV>'] for w in seq]
    tensor = torch.LongTensor(idxs)
    if host_device == "gpu":
        tensor = tensor.cuda()
    return autograd.Variable(tensor, requires_grad=requires_grad)


def log_sum_exp(vec):
    # Compute log sum exp in a numerically stable way for the forward algorithm
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class EmbeddingCRF(nn.Module):

    GPU  = "gpu"

    def __init__(self, tag_set, word_to_idx, embeddings, window=0, k=0, decay=1.0,
                 fix_embeddings=True, host_device="CPU", seed=123):
        """

        NOTE: We assume the embedding matrix and word index dicts are setup outside the model

        :param tag_set:
        :param word_to_idx:
        :param embeddings:     nxd matrix of d-dimensional embeddings
        :param fine_tune_embs: backprop errors to embedding weights
        :param host_device:    (GPU|CPU)
        :param seed:
        """
        super(EmbeddingCRF, self).__init__()

        self.seed        = seed
        self.host_device = host_device.lower()
        if not torch.cuda.is_available() and self.host_device == self.GPU:
            logger.error("Warning! CUDA not available, defaulting to CPU")
            self.host_device = "cpu"
        self._set_manual_seed()

        # feature options
        self.window = window
        self.decay  = decay
        self.k      = k

        # setup label/tag space
        self.START_TAG   = "<START>"
        self.STOP_TAG    = "<STOP>"
        self.tag_to_idx  = {t:i for i,t in enumerate(tag_set + [self.START_TAG, self.STOP_TAG])}
        self.idx_to_tag  = {i:t for t,i in self.tag_to_idx.items()}
        self.tagset_size = len(self.tag_to_idx)

        # initialize pytorch embedding matrix
        self.word_to_idx = word_to_idx
        self.emb_dim     = embeddings.shape[1]
        self.embs        = nn.Embedding(len(self.word_to_idx), self.emb_dim)
        self.embs.weight.requires_grad = fix_embeddings == False

        # maps the output of the CRF into tag space.
        # representation dimension is emb + mean(left) + mean(right)
        self.repr_dim = self.emb_dim + (self.emb_dim * (k + 1) * (2 if self.window else 0))
        self.linear2tag = nn.Linear(self.repr_dim, self.tagset_size, bias=False)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        logger.info("Using host_device: {}".format(self.host_device))
        if self.host_device == self.GPU:
            self.cuda()
            self.transitions.cuda()
            self.linear2tag.cuda()
            self.embs.cuda()

        # load embedding weights
        self.embs.weight.data.copy_(torch.from_numpy(embeddings))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_idx[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_idx[self.STOP_TAG]]  = -10000

    def _set_manual_seed(self):
        """
        Set numpy and torch manual seeds
        NOTE: GPU requires setting a seperate seed

        :return:
        """
        torch.manual_seed(self.seed)
        if self.host_device == self.GPU:
            torch.cuda.manual_seed(self.seed)

        np.random.seed(seed=int(self.seed))

    def _forward_alg(self, feats):

        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        if self.host_device == self.GPU:
            init_alphas = init_alphas.cuda()

        forward_var = autograd.Variable(init_alphas)
        init_alphas[0][self.tag_to_idx[self.START_TAG]] = 0.

        # Iterate through words/feature sets in the sentence
        for feat in feats:
            alphas_t = forward_var + self.transitions + feat.view(-1, 1).expand(feat.size()[0], self.tagset_size)
            # log_sum_exp
            max_score = torch.max(alphas_t, 1)[0]
            forward_var = (max_score + torch.log(torch.sum(torch.exp(alphas_t - max_score.view(-1, 1)), 1))).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_idx[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _forward_alg_tutorial(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_idx[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):

                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score

                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))

            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_idx[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_features(self, sentence):
        """
        Word-only embedding features
        TODO: Bring in the PCA/representation code from representation.py
        :param sentence:
        :return:
        """
        embeds = self.embs(sentence).view(len(sentence), 1, -1)
        out    = embeds.view(len(sentence), self.emb_dim)
        feats  = self.linear2tag(out)
        return feats

    def _score_sentence(self, feats, tags):
        """
        Gives the score of a provided tag sequence

        :param feats:
        :param tags:
        :return:
        """
        score = autograd.Variable(torch.Tensor([0]).cuda()) if self.host_device == self.GPU else autograd.Variable(torch.Tensor([0]))
        start_tag = torch.LongTensor([self.tag_to_idx[self.START_TAG]])
        if self.host_device == self.GPU:
            start_tag = start_tag.cuda()

        tags = torch.cat([start_tag, tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        score = score + self.transitions[self.tag_to_idx[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """

        :param feats:
        :return:
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        if self.host_device == self.GPU:
            init_vvars = init_vvars.cuda()

        init_vvars[0][self.tag_to_idx[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_idx[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        Criterion function

        :param sentence:
        :param tags:
        :return:
        """
        feats         = self._get_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score    = self._score_sentence(feats, tags)
        return forward_score - gold_score

        #forward_score = self._forward_alg(feats)
        #gold_score = self._score_sentence(feats, tags)
        #return forward_score - gold_score



    def forward(self, sentence):
        """
        TOOD: Minibatching
        :param sentence:
        :return:
        """
        feats = self._get_features(sentence)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq

    def predict(self):
        pass

class PytorchModelWrapper(object):
    """
    Wrap models to support Snorkel grid search interface
    """
    def __init__(self):
        pass

    def train(self, X_train, y_train, X_dev=None, y_dev=None, X_test=None, y_test=None, **kwargs):
        pass

    def marginals(self, X):
        pass

    def save(self):
        pass

    def load(self):
        pass

