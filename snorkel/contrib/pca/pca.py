import os
import collections
import numpy as np

import warnings

from snorkel.contrib.wclstm.sigmoid_with_binary_crossentropy import SigmoidWithBinaryCrossEntropy
from snorkel.contrib.wclstm.utils import change_marginals_with_spanset_information
from snorkel.learning.disc_learning import TFNoiseAwareModel
from snorkel.models import Candidate

from .utils import candidate_to_tokens, SymbolTable
from six.moves.cPickle import dump, load
from time import time
import scipy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from snorkel.learning.utils import reshape_marginals, LabelBalancer


class PCA(TFNoiseAwareModel):
    name = 'PCA'
    representation = True
    gpu = ['gpu', 'GPU']

    """PCA for relation extraction"""

    def gen_dist_exp(self, length, decay, st, ed):
        ret = []
        for i in range(length):
            if i < st:
                ret.append(decay ** (st - i))
            elif i >= ed:
                ret.append(decay ** (i - ed + 1))
            else:
                ret.append(1.)
        return np.array(ret)

    def gen_dist_linear(self, length, st, ed, window_size):
        ret = []
        for i in range(length):
            if i < st:
                ret.append(max(0., 1. * (window_size - st + i + 1) / window_size))
            elif i >= ed:
                ret.append(max(0., 1. * (window_size + ed - i) / window_size))
            else:
                ret.append(1.)
        return np.array(ret)

    def _get_context_seqs(self, candidates, max_window_len=5):
        """
        Given a set of candidates, generate word contexts. This is task-dependant.
        Each context c_i maps to an embedding matrix W_i in \R^{n \times d} where
        n is the # of words and d is the embedding dimension

        A) NER / 1-arity relations
           MENTION | LEFT | RIGHT | SENTENCE

        B) 2-arity relations
           MENTION_1 | MENTION_2 | M1_INNER_WORDS_M2 | SENTENCE

        We can optionally apply weighting schemes to W

        :param candidates:
        :return:
        """
        dist = collections.defaultdict(list)

        context_seqs = []
        for c in candidates:
            words    = candidate_to_tokens(c)
            m1_start = c[0].get_word_start()
            m1_end   = c[0].get_word_end() + 1

            # optional mention 1 weight decay
            if self.kernel == 'exp':
                dist_m1 = self.gen_dist_exp(len(words), self.decay, m1_start, m1_end)
                dist_sent = dist_m1
            elif self.kernel == 'linear':
                dist_m1 = self.gen_dist_linear(len(words), m1_start, m1_end, self.window_size)
                dist_m1_sent = self.gen_dist_linear(len(words), m1_start, m1_end, max(m1_start, len(words) - m1_end))
                dist_sent = dist_m1_sent

            # IGNORE FOR NOW
            # optional expand mention by window size k
            #k = self.window_size
            #m1_start = max(m1_start - k, 0)
            #m1_end = min(m1_end + k, len(words))

            # context sequences
            sent_seq  = np.array(words)
            m1_seq    = np.array(words[m1_start: m1_end])
            left_seq  = np.array(words[0: m1_start])
            right_seq = np.array(words[m1_end:])

            # enforce window constraints
            left_seq  = left_seq[-max_window_len:]
            right_seq = right_seq[:max_window_len]

            dist["S"].append(sent_seq.shape[0])
            dist["L"].append(left_seq.shape[0])
            dist["R"].append(right_seq.shape[0])
            dist["M"].append(m1_seq.shape[0])

            # use kerney decay?
            if self.kernel is None:
                context_seqs.append((sent_seq, m1_seq, left_seq, right_seq, True, c))
            else:
                # TODO -- double Check!!!!!!
                dm1 = dist_m1[m1_start: m1_end]
                #dword_seq = dist_sent[m1_start + 1: m1_end]

                dleft_seq  = dist_sent[0: m1_start]
                dright_seq = dist_sent[m1_end:]

                # enforce window constraints
                dleft_seq = dleft_seq[-max_window_len:]
                dright_seq = dright_seq[:max_window_len]

                context_seqs.append((sent_seq, m1_seq, left_seq, right_seq,
                                     True, dist_sent, dm1, dleft_seq, dright_seq))

        print("context lengths:")
        for i in dist:
            print(f"{i} {np.mean(dist[i])} {np.max(dist[i])}")

        return context_seqs

    def _preprocess_data_combination(self, candidates):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        """

        if len(candidates) > 0 and not isinstance(candidates[0], Candidate):
            return candidates

        # HACK for NER/1-arity
        if len(candidates[0]) == 1:
            return self._get_context_seqs(candidates, self.max_context_window_length)

        data = []
        for candidate in candidates:
            words = candidate_to_tokens(candidate)

            # Word level embeddings
            sent = np.array(words)

            k = self.window_size

            m1_start = candidate[0].get_word_start()
            m1_end = candidate[0].get_word_end() + 1

            if self.kernel == 'exp':
                dist_m1 = self.gen_dist_exp(len(words), self.decay, m1_start, m1_end)
            elif self.kernel == 'linear':
                dist_m1 = self.gen_dist_linear(len(words), m1_start, m1_end, self.window_size)
                dist_m1_sent = self.gen_dist_linear(len(words), m1_start, m1_end, max(m1_start, len(words) - m1_end))

            m1_start = max(m1_start - k, 0)
            m1_end = min(m1_end + k, len(words))

            m2_start = candidate[1].get_word_start()
            m2_end = candidate[1].get_word_end() + 1

            if self.kernel == 'exp':
                dist_m2 = self.gen_dist_exp(len(words), self.decay, m2_start, m2_end)
            elif self.kernel == 'linear':
                dist_m2 = self.gen_dist_linear(len(words), m2_start, m2_end, self.window_size)
                dist_m2_sent = self.gen_dist_linear(len(words), m2_start, m2_end, max(m2_start, len(words) - m2_end))

            m2_start = max(m2_start - k, 0)
            m2_end = min(m2_end + k, len(words))

            if self.kernel == 'exp':
                dist_sent = np.maximum(dist_m1, dist_m2)
            elif self.kernel == 'linear':
                dist_sent = np.maximum(dist_m1_sent, dist_m2_sent)

            m1 = np.array(words[m1_start: m1_end])
            m2 = np.array(words[m2_start: m2_end])
            st = min(candidate[0].get_word_end(), candidate[1].get_word_end())
            ed = max(candidate[0].get_word_start(), candidate[1].get_word_start())
            word_seq = np.array(words[st + 1: ed])

            order = 0 if m1_start < m2_start else 1
            if self.kernel is None:
                data.append((sent, m1, m2, word_seq, order))
            else:
                dm1 = dist_m1[m1_start: m1_end]
                dm2 = dist_m2[m2_start: m2_end]
                dword_seq = dist_sent[st + 1: ed]
                data.append((sent, m1, m2, word_seq, order, dist_sent, dm1, dm2, dword_seq))

        return data

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_sentence_length"""
        mx = max_len or self.max_sentence_length
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def _build_model(self, **model_kwargs):
        pass

    def create_dict(self, splits, word=True, char=True):
        if word: self.word_dict = SymbolTable()
        if char: self.char_dict = SymbolTable()

        for candidates in splits['train']:
            for candidate in candidates:
                words = candidate_to_tokens(candidate)
                if word: map(self.word_dict.get, words)
                if char: map(self.char_dict.get, self.gen_char_list(words, self.char_gram))
        if char:
            print("|Train Vocab|    = word:{}, char:{}".format(self.word_dict.s, self.char_dict.s))
        else:
            print("|Train Vocab|    = word:{}".format(self.word_dict.s))

        for candidates in splits['test']:
            for candidate in candidates:
                words = candidate_to_tokens(candidate)
                if word: map(self.word_dict.get, words)
                if char: map(self.char_dict.get, self.gen_char_list(words, self.char_gram))
        if char:
            print("|Total Vocab|    = word:{}, char:{}".format(self.word_dict.s, self.char_dict.s))
        else:
            print("|Total Vocab|    = word:{}".format(self.word_dict.s))

    def load_dict(self):
        # load dict from file
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
            extend_word = True
        else:
            extend_word = False

        if self.char and not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()
            extend_char = True
        else:
            extend_char = False

        if extend_word:
            # Word embeddings
            f = open(self.word_emb_path, 'r')

            l = list()
            for _ in f:
                if len(_.strip().split(' ')) > self.word_emb_dim + 1:
                    l.append(' ')
                else:
                    word = _.rstrip().split(' ')[0]
                    # Replace placeholder to original word defined by user.
                    for key in self.replace.keys():
                        word = word.replace(key, self.replace[key])
                    l.append(word)
            for w in l:
                self.word_dict.get(w)
            f.close()

        if self.char and extend_char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            l = list()
            for _ in f:
                if len(_.strip().split(' ')) > self.char_emb_dim + 1:
                    l.append(' ')
                else:
                    word = _.strip().split(' ')[0]
                    # Replace placeholder to original word defined by user.
                    for key in self.replace.keys():
                        word = word.replace(key, self.replace[key])
                    l.append(word)
            for w in l:
                self.char_dict.get(w)
            f.close()

    def load_embeddings(self):
        self.load_dict()
        # Random initial word embeddings
        self.word_emb = np.random.uniform(-0.01, 0.01, (self.word_dict.s, self.word_emb_dim)).astype(np.float)

        if self.char:
            # Random initial char embeddings
            self.char_emb = np.random.uniform(-0.01, 0.01, (self.char_dict.s, self.char_emb_dim)).astype(np.float)

        # Set unknown
        unknown_symbol = 1

        # Word embeddings
        f = open(self.word_emb_path, 'r')
        fmt = "fastText" if self.word_emb_path.split(".")[-1] == "vec" else "txt"

        for i,line in enumerate(f):

            if fmt == "fastText" and i == 0:
                continue

            line = line.rstrip().split(' ')
            if len(line) > self.word_emb_dim + 1:
                line[0] = ' '
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            if self.word_dict.lookup(line[0]) != unknown_symbol:
                self.word_emb[self.word_dict.lookup_strict(line[0])] = np.asarray(
                    [float(_) for _ in line[-self.word_emb_dim:]])
        f.close()

        if self.char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            for line in f:
                line = line.strip().split(' ')
                if len(line) > self.char_emb_dim + 1:
                    line[0] = ' '
                for key in self.replace.keys():
                    line[0] = line[0].replace(key, self.replace[key])
                if self.char_dict.lookup(line[0]) != unknown_symbol:
                    self.char_emb[self.char_dict.lookup_strict(line[0])] = np.asarray(
                        [float(_) for _ in line[-self.char_emb_dim:]])
            f.close()

    def gen_char_list(self, x, k):
        ret = []
        for w in x:
            l = ['<s>'] + list(w) + ['</s>'] * max(1, k - len(w) - 1)
            for i in range(len(l) - k + 1):
                ret.append(''.join(l[i:i + k]))
        ret = [_.replace('<s>', '') for _ in ret]
        ret = filter(lambda x: x != '', ret)
        ret = filter(lambda x: x != '</s>', ret)
        ret = [_.replace('</s>', '') + '</s>' * (k - len(_.replace('</s>', ''))) for _ in ret]
        return ret

    def get_singular_vectors(self, x):
        u, s, v = torch.svd(x)
        k = self.r if v.size(1) > self.l + self.r else v.size(1) - self.l
        z = torch.zeros(self.l + self.r, v.size(0)).double()
        z[0:self.l + k, ] = v.transpose(0, 1)[:self.l + k, ]

        if self.method in ['magic1', 'magicg']:
            theta = torch.from_numpy(self.theta[:z.size(0) * z.size(1)]).view(z.size(0), z.size(1)).double()
            z = torch.mul(z, torch.sign(torch.mul(z, theta)))

        if self.method == 'procrustes':
            r = scipy.linalg.orthogonal_procrustes(z.numpy().T, self.F[:z.size(0) * z.size(1)].reshape(z.size(0), z.size(1)).T)[0]
            z = torch.mm(torch.from_numpy(r), z)

        return z[self.l:, ]

    def get_principal_components(self, x, y=None):
        # word level features
        ret1 = torch.zeros(self.r + 1, self.word_emb_dim).double()
        if len(''.join(x)) > 0:
            f = self.word_dict.lookup
            if y is None:
                x_ = torch.from_numpy(np.array([self.word_emb[_] for _ in map(f, x)]))
            else:
                x_ = torch.from_numpy(np.diag(y).dot(np.array([self.word_emb[_] for _ in map(f, x)])))
            mu = torch.mean(x_, 0, keepdim=True)
            ret1[0, ] = mu / torch.norm(mu)
            if self.r > 0 and len(x) > self.l:
                ret1[1:, ] = self.get_singular_vectors(x_ - mu.repeat(x_.size(0), 1))

        if self.bidirectional:
            # word level features
            x_b = list(reversed(x))
            ret1_b = torch.zeros(self.r + 1, self.word_emb_dim).double()
            if len(''.join(x_b)) > 0:
                f = self.word_dict.lookup
                if y is None:
                    x_ = torch.from_numpy(np.array([self.word_emb[_] for _ in map(f, x_b)]))
                else:
                    y_b = list(reversed(y))
                    x_ = torch.from_numpy(np.diag(y_b).dot(np.array([self.word_emb[_] for _ in map(f, x_b)])))
                mu = torch.mean(x_, 0, keepdim=True)
                ret1_b[0, ] = mu / torch.norm(mu)
                if self.r > 0 and len(x_b) > self.l:
                    ret1_b[1:, ] = self.get_singular_vectors(x_ - mu.repeat(x_.size(0), 1))

        if self.char:
            # char level features
            ret2 = torch.zeros(self.r + 1, self.char_emb_dim).double()
            x_c = self.gen_char_list(x, self.char_gram)
            if len(x_c) > 0:
                f = self.char_dict.lookup
                x_ = torch.from_numpy(np.array([self.char_emb[_] for _ in map(f, list(x_c))]))
                mu = torch.mean(x_, 0, keepdim=True)
                ret2[0, ] = mu / torch.norm(mu)
                if self.r > 0 and len(x_c) > self.l:
                    ret2[1:, ] = self.get_singular_vectors(x_ - mu.repeat(x_.size(0), 1))
            if self.bidirectional:
                # char level features
                ret2_b = torch.zeros(self.r + 1, self.char_emb_dim).double()
                x_c_b = x_c[::-1]
                if len(x_c_b) > 0:
                    f = self.char_dict.lookup
                    x_ = torch.from_numpy(np.array([self.char_emb[_] for _ in map(f, list(x_c_b))]))
                    mu = torch.mean(x_, 0, keepdim=True)
                    ret2_b[0, ] = mu / torch.norm(mu)
                    if self.r > 0 and len(x_c_b) > self.l:
                        ret2_b[1:, ] = self.get_singular_vectors(x_ - mu.repeat(x_.size(0), 1))

        if self.char:
            if self.bidirectional:
                return torch.cat((ret1.view(1, -1), ret2.view(1, -1), ret1_b.view(1, -1), ret2_b.view(1, -1)), 1)
            else:
                return torch.cat((ret1.view(1, -1), ret2.view(1, -1)), 1)
        else:
            if self.bidirectional:
                return torch.cat((ret1.view(1, -1), ret1_b.view(1, -1)), 1)
            else:
                return ret1.view(1, -1)

    def _gen_ner_features(self, X):
        """
        HACK: Generate NER/1-arity PCA features

        Order of sequences is implicit in the order of the X list
        NER ordering

        [sent_seq, m1_seq, left_seq, right_seq, order]
        [sent_seq, m1_seq, left_seq, right_seq, order, dist_sent, dm1, dword_seq]

        ORIGINAL relation ordering
        [sent_seq, m1_seq, left_seq, right_seq, order, dist_sent, dm1, dleft_seq, dright_seq]

        :return:

        """
        if self.kernel is None:
            if self.sent_feat:
                sent_pca  = self.get_principal_components(X[0])
            m1_pca        = self.get_principal_components(X[1])
            if self.cont_feat:
                left_pca  = self.get_principal_components(X[2])
                right_pca = self.get_principal_components(X[3])
        else:

            if self.sent_feat:
                sent_pca  = self.get_principal_components(X[0], X[5])
            m1_pca        = self.get_principal_components(X[1], X[6])
            if self.cont_feat:
                left_pca  = self.get_principal_components(X[2], X[7])
                right_pca = self.get_principal_components(X[3], X[8])

        if self.sent_feat and self.cont_feat:
            feature = torch.cat((m1_pca, left_pca, right_pca, sent_pca))
        elif self.sent_feat:
            feature = torch.cat((m1_pca, sent_pca))
        elif self.cont_feat:
            feature = torch.cat((m1_pca, left_pca, right_pca))
        else:
            feature = torch.cat((m1_pca))

        feature = feature.view(1, -1)
        return feature

    def gen_feature(self, X):

        # HACK - NER/1-arity features
        if self.ner:
            return self._gen_ner_features(X)

        if self.kernel is None:
            m1 = self.get_principal_components(X[1])
            m2 = self.get_principal_components(X[2])
            if self.sent_feat:
                sent = self.get_principal_components(X[0])
            if self.cont_feat:
                word_seq = self.get_principal_components(X[3])
        else:
            m1 = self.get_principal_components(X[1], X[6])
            m2 = self.get_principal_components(X[2], X[7])
            if self.sent_feat:
                sent = self.get_principal_components(X[0], X[5])
            if self.cont_feat:
                word_seq = self.get_principal_components(X[3], X[8])

        if self.sent_feat and self.cont_feat:
            feature = torch.cat((m1, m2, sent, word_seq))
        elif self.sent_feat:
            feature = torch.cat((m1, m2, sent))
        elif self.cont_feat:
            feature = torch.cat((m1, m2, word_seq))
        else:
            feature = torch.cat((m1, m2))

        feature = feature.view(1, -1)

        # add indicator feature for asymmetric relation
        if self.asymmetric:
            order = torch.zeros(1, 2).double()
            if X[4] == 0:
                order[0][0] = 1.0
            else:
                order[0][1] = 1.0
            feature = torch.cat((feature, order), 1).view(1, -1)

        return feature

    def build_model(self, input_dim, output_dim):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        model = torch.nn.Sequential()
        model.add_module("linear",
                         torch.nn.Linear(input_dim, output_dim, bias=False))
        if self.host_device in self.gpu: model.cuda()
        return model

    def train_model(self, model, loss, optimizer, x_val, y_val):
        if self.host_device in self.gpu:
            x = Variable(x_val, requires_grad=False).cuda()
            y = Variable(y_val, requires_grad=False).cuda()
        else:
            x = Variable(x_val, requires_grad=False)
            y = Variable(y_val, requires_grad=False)

        # Reset gradient
        optimizer.zero_grad()

        # Forward
        fx = model.forward(x)

        if self.host_device in self.gpu:
            output = loss.forward(fx.cuda(), y)
        else:
            output = loss.forward(fx, y)

        # Backward
        output.backward()

        # Update parameters
        optimizer.step()

        return output.data[0]

    def predict(self, model, x_val):
        if self.host_device in self.gpu:
            x = Variable(x_val, requires_grad=False).cuda()
        else:
            x = Variable(x_val, requires_grad=False)
        output = model.forward(x)
        sigmoid = nn.Sigmoid()
        if self.host_device in self.gpu:
            pred = sigmoid(output).data.cpu().numpy()
        else:
            pred = sigmoid(output).data.numpy()
        return pred

    def _init_kwargs(self, **kwargs):

        self.model_kwargs = kwargs

        self.ner = kwargs.get('ner', False)

        # Set if use char embeddings
        self.char = kwargs.get('char', False)

        # Set if use whole sentence feature
        self.sent_feat = kwargs.get('sent_feat', True)

        # Set if use whole context feature
        self.cont_feat = kwargs.get('cont_feat', True)

        # Set bidirectional
        self.bidirectional = kwargs.get('bidirectional', False)

        # Set word embedding dimension
        self.word_emb_dim = kwargs.get('word_emb_dim', None)

        # Set word embedding path
        self.word_emb_path = kwargs.get('word_emb_path', None)

        # Set char gram k
        self.char_gram = kwargs.get('char_gram', 1)

        # Set char embedding dimension
        self.char_emb_dim = kwargs.get('char_emb_dim', None)

        # Set char embedding path
        self.char_emb_path = kwargs.get('char_emb_path', None)

        # Set learning rate
        self.lr = kwargs.get('lr', 1e-3)

        # Set weight decay
        self.weight_decay = kwargs.get('weight_decay', 0.)

        # Set learning epoch
        self.n_epochs = kwargs.get('n_epochs', 100)

        # Set ignore first k principal components
        self.l = kwargs.get('l', 0)

        # Set select top k principal components from the rest
        self.r = kwargs.get('r', 10)

        # Set learning batch size
        self.batch_size = kwargs.get('batch_size', 100)

        # Set rebalance setting
        self.rebalance = kwargs.get('rebalance', False)

        # Set surrounding window size for mention
        self.window_size = kwargs.get('window_size', 3)

        # Set relation type indicator (e.g., symmetric or asymmetric)
        self.asymmetric = kwargs.get('asymmetric', False)

        # Set max sentence length
        self.max_sentence_length = kwargs.get('max_sentence_length', 100)

        # Set max context window length (Left/Right window around mention)
        self.max_context_window_length = kwargs.get('max_context_window_length', 5 )

        # Set kernel
        self.kernel = kwargs.get('kernel', None)

        # Set exp
        if self.kernel == 'exp':
            self.decay = kwargs.get('decay', 1.0)

        # Set host device
        self.host_device = kwargs.get('host_device', 'cpu')

        # Set optimizer
        self.optimizer_name = kwargs.get('optimizer', 'adam')

        # Set loss
        self.loss_name = kwargs.get('loss', 'mlsml')

        # Replace placeholders in embedding files
        self.replace = kwargs.get('replace', {})

        # Set method to reduce variance
        self.method = kwargs.get('method', None)

        # Use spansets to reduce marginals
        self.use_spansets_for_marginals = kwargs.get('use_spansets_for_marginals', False)

        print("===============================================")
        print(f"Number of learning epochs:         {self.n_epochs}")
        print(f"Learning rate:                     {self.lr}")
        print(f"Ignore top l principal components: {self.l}")
        print(f"Select top k principal components: {self.r}")
        print(f"Batch size:                        {self.batch_size}")
        print(f"Rebalance:                         {self.rebalance}")
        print(f"Surrounding window size:           {self.window_size}")
        print(f"Use sentence sequence:             {self.sent_feat}")
        print(f"Use window sequence:               {self.cont_feat}")
        print(f"Bidirectional:                     {self.bidirectional}")
        print(f"Host device:                       {self.host_device}")
        print(f"Use char embeddings:               {self.char}")
        print(f"Char gram:                         {self.char_gram}")
        print(f"Kernel:                            {self.kernel}")
        if self.kernel == 'exp':
            print(f"Exp kernel decay:                  {self.decay}")
        print(f"Word embedding size:               {self.word_emb_dim}")
        print(f"Char embedding size:               {self.char_emb_dim}")
        print(f"Word embedding:                    {self.word_emb_path}")
        print(f"Char embedding:                    {self.char_emb_path}")
        print(f"Invariance method                  {self.method}")
        print(f"NER/1-arity candidates             {self.ner}")
        print("===============================================")

        assert self.word_emb_path is not None
        if self.char:
            assert self.char_emb_path is not None

        if kwargs.get('init_pretrained', False):
            self.create_dict(kwargs['init_pretrained'], word=True, char=self.char)
            del self.model_kwargs["init_pretrained"]

    def train(self, X_train, Y_train, session, X_dev=None, Y_dev=None, print_freq=5, dev_ckpt=True,
              dev_ckpt_delay=0.75, save_dir='checkpoints', **kwargs):

        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """

        print_train_scores = kwargs.get('print_train_scores', False)
        self._init_kwargs(**kwargs)

        verbose = print_freq > 0

        # Set random seed
        torch.manual_seed(self.seed)
        if self.host_device in self.gpu:
            torch.cuda.manual_seed(self.seed)

        np.random.seed(seed=int(self.seed))

        # Set random seed for all numpy operations
        self.rand_state.seed(self.seed)

        # load embeddings from file
        self.load_embeddings()

        print("Done loading embeddings...")

        if self.method == 'magic1':
            if self.char:
                self.theta = np.ones((self.l + self.r) * max(self.word_emb_dim, self.char_emb_dim))
            else:
                self.theta = np.ones((self.l + self.r) * self.word_emb_dim)
        elif self.method == 'magicg':
            if self.char:
                self.theta = np.random.normal(0, 1.0, (self.l + self.r) * max(self.word_emb_dim, self.char_emb_dim))
            else:
                self.theta = np.random.normal(0, 1.0, (self.l + self.r) * self.word_emb_dim)
        elif self.method == 'procrustes':
            self.F = None
            if self.r > 0:
                while True:
                    self.F = np.random.normal(0, 1.0, (self.l + self.r) * max(self.word_emb_dim, self.char_emb_dim))
                    f1 = self.F[:(self.l + self.r) * self.word_emb_dim].reshape(((self.l + self.r), self.word_emb_dim))
                    product1 = np.dot(f1, f1.T)
                    product1 = product1 - np.identity(product1.shape[0])
                    if self.char:
                        f2 = self.F[:(self.l + self.r) * self.char_emb_dim].reshape(((self.l + self.r), self.char_emb_dim))
                        product2 = np.dot(f2, f2.T)
                        product2 = product2 - np.identity(product2.shape[0])
                    if product1.any() == 0:
                        continue
                    if self.char and product2.any() == 0:
                        continue
                    break

        cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2
        if cardinality != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                             "match model cardinality ({1}).".format(Y_train.shape[1],
                                                                     self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)
        # Make sure marginals are in [0,1] (v.s e.g. [-1, 1])
        if self.cardinality > 2 and not np.all(Y_train.sum(axis=1) - 1 < 1e-10):
            raise ValueError("Y_train must be row-stochastic (rows sum to 1).")
        if not np.all(Y_train >= 0):
            raise ValueError("Y_train must have values in [0,1].")

        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(self.rebalance,
                                                               rand_state=self.rand_state)
        else:
            # In categorical setting, just remove unlabeled
            # diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            # train_idxs = np.where(diffs > 1e-6)[0]
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            balanced_idxs = np.where(diffs < 1e-6)[0]
            uncat_improvement = 0.05
            for i in range(self.cardinality - 1):
                Y_train[balanced_idxs, i] -= uncat_improvement / (self.cardinality - 1)
            Y_train[balanced_idxs, -1] += uncat_improvement
            if self.rebalance:
                train_idxs = LabelBalancer(Y_train, categorical=True) \
                    .rebalance_categorical_train_idxs(rebalance=self.rebalance, rand_state=self.rand_state)
            else:
                train_idxs = np.where(diffs > 0)[0]

        X_train = [X_train[j] for j in train_idxs] if self.representation \
            else X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        if verbose:
            st = time()
            print("[%s] n_train= %s" % (self.name, len(X_train)))

        X_train = self._preprocess_data_combination(X_train)
        if X_dev is not None:
            X_dev = self._preprocess_data_combination(X_dev)
        Y_train = torch.from_numpy(Y_train).float()

        new_X_train = None
        for i in range(len(X_train)):
            feature = self.gen_feature(X_train[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X_train), feature.size(1)), dtype=np.float)).float()
            new_X_train[i] = feature
        data_set = data_utils.TensorDataset(new_X_train, Y_train)
        train_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

        n_examples, n_features = new_X_train.size()
        n_classes = 1 if self.cardinality == 2 else self.cardinality

        self.model = self.build_model(n_features, n_classes)

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            warnings.warn('Couldn\'t recognize optimizer, using Adam')
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr, weight_decay=self.weight_decay)

        if self.loss_name == 'mlsml':
            loss = nn.MultiLabelSoftMarginLoss()
        elif self.loss_name == 'sbce':
            loss = SigmoidWithBinaryCrossEntropy()
        elif self.loss_name == 'bcell':
            loss = nn.BCEWithLogitsLoss()
        else:
            warnings.warn('Couldn\'t recognize loss, using MultiLabelSoftMarginLoss')
            loss = nn.MultiLabelSoftMarginLoss()


        dev_score_opt = 0.0

        for idx in range(self.n_epochs):
            cost = 0.
            for x, y in train_loader:
                cost += self.train_model(self.model, loss, optimizer, x, y.float())
            if verbose and ((idx + 1) % print_freq == 0 or idx + 1 == self.n_epochs):
                print(f'Finished learning in epoch {idx + 1}')
                msg = "[%s] Epoch %s, Training error: %s" % (self.name, idx + 1, cost)
                score_label = "F1"
                if print_train_scores:
                    if cardinality == 2:
                        Y_train[Y_train > 0.5] = 1
                        Y_train[Y_train <= 0.5] = 0
                        train_scores = self.score(X_train, Y_train, batch_size=self.batch_size)
                        train_score = train_scores[-1]
                    else:
                        print('Calculating train scores...')
                        train_scores = self.error_analysis(session, [x[-1] for x in X_train],
                                                           (Y_train.max(dim=1)[1] + 1) % self.cardinality,
                                                           display=True,
                                                           batch_size=self.batch_size)
                        train_score = train_scores[2]
                    msg += '\tTrain {0}={1:.2f}'.format(score_label, 100. * train_score)
                if X_dev is not None:
                    print('Calculating dev scores...')
                    dev_scores = self.error_analysis(session, [x[-1] for x in X_dev], Y_dev,
                                                     batch_size=self.batch_size)
                    dev_score = dev_scores[2]

                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * dev_score)
                print(msg)

                if X_dev is not None and dev_ckpt and idx > dev_ckpt_delay * self.n_epochs and dev_score > dev_score_opt:
                    dev_score_opt = dev_score
                    self.save(save_dir=save_dir, only_param=True)

        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time() - st))

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.load(save_dir=save_dir, only_param=True)

    def _marginals_batch(self, X):
        new_X_train = None
        X = self._preprocess_data_combination(X)

        for i in range(len(X)):
            feature = self.gen_feature(X[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X), feature.size(1))))
            new_X_train[i] = feature

        new_X_train = new_X_train.float()

        all_marginals = self.predict(self.model, new_X_train)

        if self.use_spansets_for_marginals:
            change_marginals_with_spanset_information([x[-1] for x in X], all_marginals)
        return all_marginals

    def embed(self, X):

        new_X_train = None
        X = self._preprocess_data_combination(X)

        for i in range(len(X)):
            feature = self.gen_feature(X[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X), feature.size(1))))
            new_X_train[i] = feature

        return new_X_train.float().numpy()

    def save(self, model_name=None, save_dir='checkpoints', verbose=True,
             only_param=False):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not only_param:
            # Save model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
                dump(self.model_kwargs, f)

            # Save model dicts needed to rebuild model
            with open(os.path.join(model_dir, "model_dicts.pkl"), 'wb') as f:
                if self.char:
                    dump({'char_dict': self.char_dict, 'word_dict': self.word_dict, 'char_emb': self.char_emb,
                          'word_emb': self.word_emb}, f)
                else:
                    dump({'word_dict': self.word_dict, 'word_emb': self.word_emb}, f)

            if self.method:
                # Save model invariance data needed to rebuild model
                with open(os.path.join(model_dir, "model_invariance.pkl"), 'wb') as f:
                    if self.method in ['magic1', 'magicg']:
                        dump({'theta': self.theta}, f)
                    else:
                        dump({'F': self.F}, f)

        torch.save(self.model, os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Model saved as <{1}>, only_param={2}".format(self.name, model_name, only_param))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True, only_param=False):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        if not only_param:
            # Load model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
                model_kwargs = load(f)
                self._init_kwargs(**model_kwargs)

            # Save model dicts needed to rebuild model
            with open(os.path.join(model_dir, "model_dicts.pkl"), 'rb') as f:
                d = load(f)
                self.word_dict = d['word_dict']
                self.word_emb = d['word_emb']
                if self.char:
                    self.char_dict = d['char_dict']
                    self.char_emb = d['char_emb']

            if self.method:
                # Save model invariance data needed to rebuild model
                with open(os.path.join(model_dir, "model_invariance.pkl"), 'rb') as f:
                    d = load(f)
                    if self.method in ['magic1', 'magicg']:
                        self.theta = d['theta']
                    else:
                        self.F = d['F']

        self.model = torch.load(os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Loaded model <{1}>, only_param={2}".format(self.name, model_name, only_param))
