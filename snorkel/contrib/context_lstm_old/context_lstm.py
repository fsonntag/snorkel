import os
import warnings
from itertools import chain
from time import time

import torch.utils.data as data_utils
from six.moves.cPickle import dump, load
from torch.optim.lr_scheduler import ReduceLROnPlateau

from snorkel.contrib.context_lstm_old.layers import *
from snorkel.contrib.context_lstm_old.sigmoid_with_binary_crossentropy import SigmoidWithBinaryCrossEntropy
from snorkel.contrib.context_lstm_old.utils import *
from snorkel.learning.spanset_classifier import SpansetClassifier
from snorkel.learning.utils import reshape_marginals, LabelBalancer


class ContextLSTM(SpansetClassifier):
    name = 'ContextLSTMOld'
    representation = True
    char_marker = ['<', '>']
    gpu = ['gpu', 'GPU']

    # Set unknown
    unknown_symbol = 1

    """Hierarchy Bi-LSTM for entity extraction"""

    def __init__(self, n_threads=None, seed=123, **kwargs):
        self.n_threads = n_threads
        self.seed = seed
        self.rand_state = np.random.RandomState()
        super(ContextLSTM, self).__init__(**kwargs)

    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
            # Add paddings for words
            for padding in ['~~[[1', '1]]~~', '~~[[2', '2]]~~']:
                self.word_dict.get(padding)

        if not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()
            # Add paddings for chars
            for marker in self.char_marker:
                self.char_dict.get(marker)

        context_word_seq_data = []
        candidate_word_seq_data = []
        candidate_char_seq_data = []
        for i, candidate in enumerate(candidates):
            # Mark sentence based on cardinality of relation
            if len(candidate) == 2:
                args = [
                    (candidate[0].get_word_start(), candidate[0].get_word_end(), 1),
                    (candidate[1].get_word_start(), candidate[1].get_word_end(), 2)
                ]
            else:
                args = [(candidate[0].get_word_start(), candidate[0].get_word_end(), 1)]

            s = trim_with_radius(mark_sentence(candidate_to_tokens(candidate), args), candidate, self.context_radius)
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            context_word_seq_data.append(np.array(list(map(f, s))))

            candidate_tokens = candidate[0].get_attrib_tokens('words')
            candidate_word_seq_data.append(np.array(list(map(f, candidate_tokens))))

            # Either extend char table or retrieve from it
            g = self.char_dict.get if extend else self.char_dict.lookup
            char_seq = []
            for w in candidate_tokens:
                word = self.char_marker[0] + w + self.char_marker[1]
                word_char_seq = []
                for i in range(len(word) - self.char_gram + 1):
                    word_char_seq.append(word[i:i + self.char_gram])
                char_seq.append(np.array(list(map(g, word_char_seq))))
            candidate_char_seq_data.append(char_seq)
        return np.array(context_word_seq_data), np.array(candidate_word_seq_data), np.array(candidate_char_seq_data)

    def create_dict(self, splits, word=True, char=True):
        """Create global dict from user input"""
        if word:
            self.word_dict = SymbolTable()
            self.word_dict_all = {}

            # Add paddings for words
            for padding in ['~~[[1', '1]]~~', '~~[[2', '2]]~~']:
                self.word_dict.get(padding)

        if char:
            self.char_dict = SymbolTable()
            self.char_dict_all = {}

            # Add paddings for chars
            for marker in self.char_marker:
                self.char_dict_all.get(marker)

        # Initialize training vocabulary
        for candidate in splits["train"]:
            words = candidate_to_tokens(candidate)
            if word:
                for w in words:
                    self.word_dict.get(w)
            if char:
                for c in list(' '.join(words)):
                    self.char_dict.get(c)

        # Initialize pre-trained vocabulary
        for candset in splits["test"]:
            for candidate in candset:
                words = candidate_to_tokens(candidate)
                if word:
                    self.word_dict_all.update(dict.fromkeys(words))
                if word:
                    self.char_dict_all.update(dict.fromkeys(list(' '.join(words))))

        print("|Train Vocab|    = {}".format(self.word_dict.s))
        print("|Dev/Test Vocab| = {}".format(len(self.word_dict_all)))

    def load_char_dict(self):
        """Load dict from user input embeddings"""
        if not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()

            # Add paddings for chars
        for marker in self.char_marker:
            self.char_dict.get(marker)

        # Char embeddings
        f = open(self.char_emb_path, 'r')

        l = list()
        for _ in f:
            line = _.strip().split(' ')
            assert (self.char_emb_dim + 1 == len(line)), "Char embedding dimension doesn't match!"
            char = line[0]
            # Replace placeholder to original word defined by user.
            for key in self.replace.keys():
                char = char.replace(key, self.replace[key])
            if hasattr(self, 'char_dict_all') and char in self.char_dict_all:
                l.append(char)

        for char in l:
            self.char_dict.get(char)
        f.close()

    def load_word_dict(self):
        """Load dict from user input embeddings"""
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()

        # Add paddings
        for padding in ['~~[[1', '1]]~~', '~~[[2', '2]]~~']:
            self.word_dict.get(padding)

        # Word embeddings
        f = open(self.word_emb_path, 'r')
        fmt = "fastText" if self.word_emb_path.split(".")[-1] == "vec" else "txt"

        n, N = 0.0, 0.0

        l = list()
        for i, _ in enumerate(f):
            if fmt == "fastText" and i == 0: continue
            line = _.rstrip().split(' ')
            assert (len(line) == self.word_emb_dim + 1), "Word embedding dimension doesn't match!"
            word = line[0]
            # Replace placeholder to original word defined by user.
            for key in self.replace.keys():
                word = word.replace(key, self.replace[key])
            if hasattr(self, 'word_dict_all') and word in self.word_dict_all:
                l.append(word)
                n += 1

        for w in l:
            self.word_dict.get(w)
        if hasattr(self, 'word_dict_all'):
            N = len(self.word_dict_all)
            print("|Dev/Test Vocab|                   = {}".format(N))
            print("|Dev/Test Vocab ^ Pretrained Embs| = {} {:2.2f}%".format(n, n / float(N) * 100))
            print("|Vocab|                            = {}".format(self.word_dict.s))
        f.close()

    def load_char_embeddings(self):
        """Load pre-trained embeddings from user input"""
        self.load_char_dict()

        # Random initial char embeddings
        self.char_emb = np.random.uniform(-0.1, 0.1, (self.char_dict.s, self.char_emb_dim)).astype(np.float)

        # Char embeddings
        f = open(self.char_emb_path, 'r')

        for line in f:
            line = line.strip().split(' ')
            assert (len(line) == self.char_emb_dim + 1), "Char embedding dimension doesn't match!"
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            if self.char_dict.lookup(line[0]) != self.unknown_symbol:
                self.char_emb[self.char_dict.lookup_strict(line[0])] = np.asarray(
                    [float(_) for _ in line[-self.char_emb_dim:]])
        f.close()

    def load_word_embeddings(self):
        """Load pre-trained embeddings from user input"""
        self.load_word_dict()
        # Random initial word embeddings
        self.word_emb = np.random.uniform(-0.1, 0.1, (self.word_dict.s, self.word_emb_dim)).astype(np.float)

        # Word embeddings
        f = open(self.word_emb_path, 'r')
        fmt = "fastText" if self.word_emb_path.split(".")[-1] == "vec" else "txt"

        for i, line in enumerate(f):
            if fmt == "fastText" and i == 0:
                continue
            line = line.rstrip().split(' ')
            assert (len(line) == self.word_emb_dim + 1), "Word embedding dimension doesn't match!"
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            if self.word_dict.lookup(line[0]) != self.unknown_symbol:
                self.word_emb[self.word_dict.lookup_strict(line[0])] = np.asarray(
                    [float(_) for _ in line[-self.word_emb_dim:]])
        f.close()

    def train_model(self, w_model, c_model, optimizer, criterion,
                    context_x_w, context_x_w_mask,
                    candidate_x_w, candidate_x_w_mask,
                    x_c, x_c_mask, y):
        """Train LSTM model"""
        w_model.train()
        c_model.train()
        batch_size, max_sent, max_token = x_c.size()
        context_w_state_word, candidate_w_state_word = w_model.init_hidden(batch_size)
        c_state_word = c_model.init_hidden(batch_size)

        if self.host_device in self.gpu:
            context_x_w = context_x_w.cuda()
            context_x_w_mask = context_x_w_mask.cuda()
            candidate_x_w = candidate_x_w.cuda()
            candidate_x_w_mask = candidate_x_w_mask.cuda()
            x_c = x_c.cuda()
            x_c_mask = x_c_mask.cuda()
            y = y.cuda()
            context_w_state_word = (context_w_state_word[0].cuda(), context_w_state_word[1].cuda())
            candidate_w_state_word = (candidate_w_state_word[0].cuda(), candidate_w_state_word[1].cuda())
            c_state_word = (c_state_word[0].cuda(), c_state_word[1].cuda())

        optimizer.zero_grad()
        s = None
        for i in range(max_sent):
            _s = c_model(x_c[:, i, :], x_c_mask[:, i, :], c_state_word)
            _s = _s.unsqueeze(0)
            s = _s if s is None else torch.cat((s, _s), 0)
        s = s.transpose(0, 1)
        y_pred, _ = w_model(context_x_w, context_x_w_mask, context_w_state_word,
                            candidate_x_w, candidate_x_w_mask, candidate_w_state_word,
                            s)

        if self.host_device in self.gpu:
            loss = criterion(y_pred.squeeze(1).cuda(), y)
        else:
            loss = criterion(y_pred.squeeze(1), y)

        loss.backward()
        optimizer.step()
        return loss.data[0]

    def _init_kwargs(self, **kwargs):
        """Parse user input arguments"""
        self.model_kwargs = kwargs

        if kwargs.get('init_pretrained', False):
            self.create_dict(kwargs['init_pretrained'], word=True, char=True)

        # Set use pre-trained embedding or not
        self.load_word_emb = kwargs.get('load_word_emb', False)
        self.load_char_emb = kwargs.get('load_char_emb', False)

        # Set word embedding dimension
        self.word_emb_dim = kwargs.get('word_emb_dim', 300)
        # Set char embedding dimension
        self.char_emb_dim = kwargs.get('char_emb_dim', 300)

        # Set word embedding path
        self.word_emb_path = kwargs.get('word_emb_path', None)
        # Set char embedding path
        self.char_emb_path = kwargs.get('char_emb_path', None)

        # Set use xavier initialization for LSTM
        self.use_xavier_init_lstm = kwargs.get('use_xavier_init_lstm', False)

        # Set shuffle batch data
        self.shuffle = kwargs.get('shuffle', False)

        # Set use attention or not
        self.attention = kwargs.get('attention', True)

        # Set learning epoch
        self.n_epochs = kwargs.get('n_epochs', 100)

        # Set dropout
        self.dropout = kwargs.get('dropout', 0.0)

        # Set learning batch size
        self.batch_size = kwargs.get('batch_size', 100)

        # Set rebalance setting
        self.rebalance = kwargs.get('rebalance', False)

        # Set char gram k
        self.char_gram = kwargs.get('char_gram', 3)

        # Set lstm hidden dimension
        self.lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 100)

        # Set bidirectional or not
        self.bidirectional = kwargs.get('bidirectional', True)

        # Set max sentence length
        self.context_radius = kwargs.get('context_radius', 10)

        # Set max word length
        self.max_word_length = kwargs.get('max_word_length', 20)

        # Set host device
        self.host_device = kwargs.get('host_device', 'cpu')

        # Set optimizer
        self.optimizer_name = kwargs.get('optimizer', 'adam')

        # Set optimizer kwargs
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', {'lr': 1e-3})

        # Set scheduler
        self.use_scheduler = kwargs.get('use_scheduler', False)

        # Use F1 score for scheduler. Otherwise use loss
        self.use_f1_for_scheduler = kwargs.get('use_f1_for_scheduler', False)

        # Scheduler kwargs
        self.scheduler_kwargs = kwargs.get('scheduler_kwargs', {})

        # Set loss
        self.loss_name = kwargs.get('loss', 'mlsml')

        # Replace placeholders in embedding files
        self.replace = kwargs.get('replace', {})

        # Set patience (number of epochs to wait without model improvement)
        self.patience = kwargs.get('patience', 100)

        print("===============================================")
        print(f"Number of learning epochs:     {self.n_epochs}")
        print(f"Optimizer:                     {self.optimizer_name}")
        print(f"Optimizer kwargs:              {self.optimizer_kwargs}")
        print(f"Use scheduler:                 {self.use_scheduler}")
        print(f"Use F1 for scheduler:          {self.use_f1_for_scheduler}")
        print(f"Scheduler kwargs:              {self.scheduler_kwargs}")
        print(f"Loss:                          {self.loss_name}")
        print(f"Use attention:                 {self.attention}")
        print(f"LSTM hidden dimension:         {self.lstm_hidden_dim}")
        print(f"dropout:                       {self.dropout}")
        print(f"Batch size:                    {self.batch_size}")
        print(f"Rebalance:                     {self.rebalance}")
        print(f"Checkpoint Patience:           {self.patience}")
        print(f"Char gram:                     {self.char_gram}")
        print(f"Max word length:               {self.max_word_length}")
        print(f"Context radius:                {self.context_radius}")
        print(f"Load pre-trained word emb.:    {self.load_word_emb}")
        print(f"Load pre-trained char emb.:    {self.load_char_emb}")
        print(f"Host device:                   {self.host_device}")
        print(f"Word embedding size:           {self.word_emb_dim}")
        print(f"Char embedding size:           {self.char_emb_dim}")
        print(f"Word embedding:                {self.word_emb_path}")
        print(f"Char embedding:                {self.char_emb_path}")
        print("===============================================")

        if self.load_word_emb:
            assert self.word_emb_path is not None
        if self.load_char_emb:
            assert self.char_emb_path is not None

        if "init_pretrained" in kwargs:
            del self.model_kwargs["init_pretrained"]

    def train(self, X_train, Y_train, session, X_dev=None, Y_dev=None, gold_candidate_set=None, print_freq=5,
              dev_ckpt=True, dev_ckpt_delay=0.25, save_dir='checkpoints', print_train_scores=False, **kwargs):

        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """

        self._init_kwargs(**kwargs)

        verbose = print_freq > 0

        # Set random seed
        torch.manual_seed(self.seed)
        if self.host_device in self.gpu:
            torch.cuda.manual_seed(self.seed)

        np.random.seed(seed=int(self.seed))

        # Set random seed for all numpy operations
        self.rand_state.seed(self.seed)

        cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2
        if cardinality != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not "
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
            if self.rebalance:
                train_idxs = LabelBalancer(Y_train, categorical=True) \
                    .rebalance_categorical_train_idxs(rebalance=self.rebalance, rand_state=self.rand_state)
            else:
                diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
                train_idxs = np.where(diffs > 0)[0]

        X_train = [X_train[j] for j in train_idxs] if self.representation \
            else X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        if verbose:
            st = time()
            print("[%s] n_train = %s" % (self.name, len(X_train)))

        context_X_w_train, candidate_X_w_train, candidate_X_c_train = \
            self._preprocess_data(X_train, extend=True)
        X_train_transformed = context_X_w_train, candidate_X_w_train, candidate_X_c_train

        X_dev_transformed = self._preprocess_data(X_dev, extend=True)

        if self.load_char_emb:
            self.load_char_embeddings()

        if self.load_word_emb:
            # load embeddings from file
            self.load_word_embeddings()

            print("Done loading pre-trained embeddings...")

        Y_train = torch.from_numpy(Y_train).float()

        X = torch.from_numpy(np.arange(len(context_X_w_train)))
        data_set = data_utils.TensorDataset(X, Y_train)
        data_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=self.shuffle)

        n_classes = 1 if self.cardinality == 2 else self.cardinality

        self.candidate_char_model = CharRNN(batch_size=self.batch_size, num_tokens=self.char_dict.s,
                                            embed_size=self.char_emb_dim,
                                            lstm_hidden=self.lstm_hidden_dim,
                                            attention=self.attention,
                                            dropout=self.dropout,
                                            bidirectional=self.bidirectional,
                                            use_cuda=self.host_device in self.gpu)
        if self.load_char_emb:
            # Set pre-trained embedding weights
            self.candidate_char_model.lookup.weight.data.copy_(torch.from_numpy(self.char_emb))

        b = 2 if self.bidirectional else 1
        self.combined_word_model = CombinedRNN(n_classes=n_classes, batch_size=self.batch_size,
                                               num_word_tokens=self.word_dict.s,
                                               embed_size=self.word_emb_dim,
                                               candidate_input_size=self.word_emb_dim + b * self.lstm_hidden_dim,
                                               lstm_hidden=self.lstm_hidden_dim,
                                               attention=self.attention,
                                               dropout=self.dropout,
                                               bidirectional=self.bidirectional,
                                               use_cuda=self.host_device in self.gpu)

        if self.load_word_emb:
            # Set pre-trained embedding weights
            self.combined_word_model.word_lookup.weight.data.copy_(torch.from_numpy(self.word_emb))

        if self.host_device in self.gpu:
            self.candidate_char_model.cuda()
            self.combined_word_model.cuda()

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                list(self.candidate_char_model.parameters()) + list(self.combined_word_model.parameters()),
                **self.optimizer_kwargs)
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                list(self.candidate_char_model.parameters()) + list(self.combined_word_model.parameters()),
                **self.optimizer_kwargs)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                list(self.candidate_char_model.parameters()) + list(self.combined_word_model.parameters()),
                **self.optimizer_kwargs)
        else:
            warnings.warn('Couldn\'t recognize optimizer, using Adam')
            optimizer = torch.optim.Adam(
                list(self.candidate_char_model.parameters()) + list(self.combined_word_model.parameters()),
                **self.optimizer_kwargs)

        if self.use_scheduler:
            if self.use_f1_for_scheduler:
                self.scheduler_kwargs['mode'] = 'max'
            scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)

        if self.loss_name == 'mlsml':
            loss = nn.MultiLabelSoftMarginLoss()
        elif self.loss_name == 'sbce':
            loss = SigmoidWithBinaryCrossEntropy()
        elif self.loss_name == 'bcell':
            loss = nn.BCEWithLogitsLoss()
        else:
            warnings.warn('Couldn\'t recognize loss, using MultiLabelSoftMarginLoss')
            loss = nn.MultiLabelSoftMarginLoss()

        last_epoch_opt = None

        for idx in range(self.n_epochs):
            cost = 0.
            for x, y in data_loader:
                context_x_w, context_x_w_mask, candidate_x_w, candidate_x_w_mask, x_c, x_c_mask = \
                    pad_batch(context_X_w_train[x.numpy()], candidate_X_w_train[x.numpy()],
                              candidate_X_c_train[x.numpy()], self.context_radius, self.max_word_length)
                y = Variable(y.float(), requires_grad=False)
                cost += self.train_model(self.combined_word_model, self.candidate_char_model, optimizer, loss,
                                         context_x_w, context_x_w_mask,
                                         candidate_x_w, candidate_x_w_mask,
                                         x_c, x_c_mask, y)
            self.cost_history.append((idx, cost))
            if verbose and ((idx + 1) % print_freq == 0 or idx + 1 == self.n_epochs):
                print(f'Finished learning in epoch {idx + 1}')
                msg = "[{}] Epoch {}, Training error: {:.2f}".format(self.name, idx + 1, cost)
                score_label = "F1"
                if print_train_scores:
                    if cardinality == 2:
                        Y_train[Y_train > 0.5] = 1
                        Y_train[Y_train <= 0.5] = 0
                        train_scores = self.score(X_train, Y_train, batch_size=self.batch_size)
                        train_score = train_scores[-1]
                    else:
                        print('Calculating train scores...')
                        train_scores = self.spanset_error_analysis(session, X_train,
                                                                   X_train_transformed,
                                                                   Y_train.numpy(),
                                                                   display=True,
                                                                   batch_size=self.batch_size,
                                                                   prediction_type='train')
                        train_score = train_scores[2]
                    self.train_history.append((idx, train_score))
                    msg += '\tTrain {0}={1:.2f}'.format(score_label, 100. * train_score)
                if X_dev is not None:
                    print('Calculating dev scores...')
                    dev_scores = self.spanset_error_analysis(session, X_dev,
                                                             X_dev_transformed,
                                                             Y_dev,
                                                             gold_candidate_set,
                                                             batch_size=self.batch_size,
                                                             prediction_type='dev')
                    dev_score = dev_scores[2]
                    self.dev_history.append((idx, dev_score))
                    msg += '\tDev {0}={1:.2f}'.format(score_label, 100. * dev_score)
                print(msg)

                if X_dev is not None and dev_ckpt and idx + 1 > dev_ckpt_delay * self.n_epochs and dev_score > self.dev_score_opt:
                    self.dev_score_opt = dev_score
                    self.dev_scores_opt = dev_scores[:3]
                    self.save(save_dir=save_dir, only_param=True)
                    last_epoch_opt = idx

                if last_epoch_opt is not None and (idx - last_epoch_opt > self.patience) and (
                        dev_ckpt and idx > dev_ckpt_delay * self.n_epochs):
                    print("[{}] No model improvement after {} epochs, halting".format(self.name, idx - last_epoch_opt))
                    break

                if self.use_scheduler:
                    if self.use_f1_for_scheduler \
                            and verbose and ((idx + 1) % print_freq == 0 or idx + 1 == self.n_epochs):
                        scheduler.step(dev_score, idx)
                    else:
                        scheduler.step(loss, idx)

        # Conclude training
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time() - st))

        self.write_history()

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and self.dev_score_opt > 0:
            self.load(save_dir=save_dir, only_param=True)

    def _marginals_batch(self, X, save_attentions=False):
        """Predict class based on user input"""
        self.candidate_char_model.eval()
        self.combined_word_model.eval()

        context_X_w, candidate_X_w, candidate_X_c = X[0], X[1], X[2]
        sigmoid = nn.Sigmoid()

        y = np.array([])
        if self.cardinality > 2:
            y = y.reshape(0, self.cardinality)

        x = torch.from_numpy(np.arange(len(context_X_w)))
        data_set = data_utils.TensorDataset(x, x)
        data_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)
        context_attentions = []
        candidate_attention = []

        for x, _ in data_loader:
            context_x_w, context_x_w_mask, candidate_x_w, candidate_x_w_mask, x_c, x_c_mask \
                = pad_batch(context_X_w[x.numpy()], candidate_X_w[x.numpy()], candidate_X_c[x.numpy()],
                            self.context_radius, self.max_word_length)
            batch_size, max_sent, max_token = x_c.size()
            context_w_state_word, candidate_w_state_word = self.combined_word_model.init_hidden(batch_size)
            c_state_word = self.candidate_char_model.init_hidden(batch_size)

            if self.host_device in self.gpu:
                context_x_w = context_x_w.cuda()
                context_x_w_mask = context_x_w_mask.cuda()
                candidate_x_w = candidate_x_w.cuda()
                candidate_x_w_mask = candidate_x_w_mask.cuda()
                x_c = x_c.cuda()
                x_c_mask = x_c_mask.cuda()
                context_w_state_word = (context_w_state_word[0].cuda(), context_w_state_word[1].cuda())
                candidate_w_state_word = (candidate_w_state_word[0].cuda(), candidate_w_state_word[1].cuda())
                c_state_word = (c_state_word[0].cuda(), c_state_word[1].cuda())

            s = None
            for i in range(max_sent):
                _s = self.candidate_char_model(x_c[:, i, :], x_c_mask[:, i, :], c_state_word)
                _s = _s.unsqueeze(0)
                s = _s if s is None else torch.cat((s, _s), 0)
            s = s.transpose(0, 1)
            y_pred, (batch_context_attentions, batch_candidate_attentions) = \
                self.combined_word_model(context_x_w,
                                         context_x_w_mask,
                                         context_w_state_word,
                                         candidate_x_w,
                                         candidate_x_w_mask,
                                         candidate_w_state_word,
                                         s)
            if self.host_device in self.gpu:
                if self.cardinality > 2:
                    y = np.vstack((y, sigmoid(y_pred).data.cpu().numpy()))
                else:
                    y = np.append(y, sigmoid(y_pred).data.cpu().numpy())
                if save_attentions:
                    context_attentions.append(batch_context_attentions.data.cpu().numpy())
                    candidate_attention.append(batch_candidate_attentions.data.cpu().numpy())
            else:
                if self.cardinality > 2:
                    y = np.vstack((y, sigmoid(y_pred).data.numpy()))
                else:
                    y = np.append(y, sigmoid(y_pred).data.numpy())
                if save_attentions:
                    context_attentions.append(batch_context_attentions.data.numpy())
                    candidate_attention.append(batch_candidate_attentions.data.numpy())
        if save_attentions:
            return y, (context_attentions, candidate_attention)
        else:
            return y

    def marginals(self, X, batch_size=None, **kwargs):
        """
        Compute the marginals for the given candidates X.
        Split into batches to avoid OOM errors, then call _marginals_batch;
        defaults to no batching.
        """
        if batch_size is None:
            all_marginals = self._marginals_batch(X)
        else:
            N = len(X[0]) if self.representation else X[0].shape[0]

            # Iterate over batches
            batch_marginals = []
            for b in range(0, N, batch_size):
                batch = self._marginals_batch((X[0][b:b + batch_size], X[1][b:b + batch_size], X[2][b:b + batch_size]))
                # Note: Make sure a list is returned!
                if min(b + batch_size, N) - b == 1:
                    batch = np.array([batch])
                batch_marginals.append(batch)
            all_marginals = np.concatenate(batch_marginals)
        return all_marginals

    def marginals_with_attention(self, X_candidates, X, batch_size=None, **kwargs):
        """
        Compute the marginals for the given candidates X.
        Split into batches to avoid OOM errors, then call _marginals_batch;
        defaults to no batching.
        Additionally print the attention weights
        """
        if batch_size is None:
            all_marginals, (context_weights, candidate_weights) = self._marginals_batch(X, save_attentions=True)
        else:
            N = len(X[0]) if self.representation else X[0].shape[0]

            # Iterate over batches
            batch_marginals = []
            context_weights, candidate_weights = [], []
            for b in range(0, N, batch_size):
                batch, (batch_context_weights, batch_candidate_weights) = self._marginals_batch(
                    (X[0][b:b + batch_size], X[1][b:b + batch_size], X[2][b:b + batch_size]), save_attentions=True)
                # Note: Make sure a list is returned!
                if min(b + batch_size, N) - b == 1:
                    batch = np.array([batch])
                batch_marginals.append(batch)
                context_weights.append(batch_context_weights)
                candidate_weights.append(batch_candidate_weights)
            all_marginals = np.concatenate(batch_marginals)
            context_weights = list(chain.from_iterable(context_weights))
            candidate_weights = list(chain.from_iterable(candidate_weights))
        max_context_length = max(a.shape[1] for a in context_weights)
        context_weights = np.concatenate(
            [np.pad(a[:, :, 0], ((0, 0), (0, max_context_length - a.shape[1])), "constant",
                    constant_values=(-1e12, -1e12))
             for a
             in context_weights])
        max_candidate_length = max(a.shape[1] for a in candidate_weights)
        candidate_weights = np.concatenate(
            [np.pad(a[:, :, 0], ((0, 0), (0, max_candidate_length - a.shape[1])), "constant",
                    constant_values=(-1e12, -1e12))
             for a
             in candidate_weights])
        write_attention(X_candidates, context_weights, candidate_weights, self.context_radius, self.output_path)

        return all_marginals

    def save(self, model_name=None, save_dir='checkpoints', verbose=True, only_param=False):
        """Save current model"""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not only_param:
            # Save model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
                dump(self.model_kwargs, f)

            if self.load_char_emb:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "char_model_dicts.pkl"), 'wb') as f:
                    dump({'char_dict': self.char_dict, 'char_emb': self.char_emb}, f)
            else:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "char_model_dicts.pkl"), 'wb') as f:
                    dump({'char_dict': self.char_dict}, f)

            if self.load_word_emb:
                with open(os.path.join(model_dir, "word_model_dicts.pkl"), 'wb') as f:
                    dump({'word_dict': self.word_dict, 'word_emb': self.word_emb}, f)
            else:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "word_model_dicts.pkl"), 'wb') as f:
                    dump({'word_dict': self.word_dict}, f)

        torch.save(self.combined_word_model, os.path.join(model_dir, model_name + '_word_model'))
        torch.save(self.candidate_char_model, os.path.join(model_dir, model_name + '_char_model'))

        if verbose:
            print("[{0}] Model saved as <{1}>, only_param={2}".format(self.name, model_name, only_param))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True, only_param=False, host_device='cpu'):
        """Load model from file and rebuild in new model"""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        if not only_param:
            # Load model kwargs needed to rebuild model
            with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
                model_kwargs = load(f)
                model_kwargs['host_device'] = host_device
                self._init_kwargs(**model_kwargs)

            if self.load_char_emb:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "char_model_dicts.pkl"), 'rb') as f:
                    d = load(f)
                    self.char_dict = d['char_dict']
                    self.char_emb = d['char_emb']
            else:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "char_model_dicts.pkl"), 'rb') as f:
                    d = load(f)
                    self.char_dict = d['char_dict']
            if self.load_word_emb:
                with open(os.path.join(model_dir, "word_model_dicts.pkl"), 'rb') as f:
                    d = load(f)
                    self.word_dict = d['word_dict']
                    self.word_emb = d['word_emb']
            else:
                # Save model dicts needed to rebuild model
                with open(os.path.join(model_dir, "word_model_dicts.pkl"), 'rb') as f:
                    d = load(f)
                    self.word_dict = d['word_dict']

        if self.host_device in self.gpu:
            self.combined_word_model = torch.load(os.path.join(model_dir, model_name + '_word_model'))
            self.candidate_char_model = torch.load(os.path.join(model_dir, model_name + '_char_model'))
            self.combined_word_model.cuda()
            self.candidate_char_model.cuda()
        else:
            self.combined_word_model = torch.load(os.path.join(model_dir, model_name + '_word_model'),
                                                  lambda storage, loc: storage)
            self.candidate_char_model = torch.load(os.path.join(model_dir, model_name + '_char_model'),
                                                   lambda storage, loc: storage)

        if verbose:
            print("[{0}] Loaded model <{1}>, only_param={2}".format(self.name, model_name, only_param))
