import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, batch_size, num_tokens, embed_size, lstm_hidden, dropout=0.0, attention=True, bidirectional=True,
                 use_cuda=False, use_xavier_init_lstm=False):

        super(CharRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.attention = attention
        self.use_cuda = use_cuda

        self.drop = nn.Dropout(dropout)
        self.lookup = nn.Embedding(num_tokens, embed_size, padding_idx=0)

        b = 2 if self.bidirectional else 1

        self.char_lstm = nn.LSTM(embed_size, lstm_hidden, batch_first=True, dropout=dropout,
                                 bidirectional=self.bidirectional)
        if use_xavier_init_lstm:
            for weights in self.char_lstm.all_weights:
                for weight in weights:
                    if len(weight.data.shape) == 2:
                        nn.init.xavier_uniform(weight, gain=math.sqrt(2.0))
        if attention:
            self.attn_linear_w_1 = nn.Linear(b * lstm_hidden, b * lstm_hidden, bias=True)
            self.attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)

    def forward(self, x, x_mask, c_state):
        """
        x      : batch_size * length
        x_mask : batch_size * length
        """
        x_emb = self.drop(self.lookup(x))
        output_char, c_state = self.char_lstm(x_emb, c_state)
        output_char = self.drop(output_char)
        if self.attention:
            """
            An attention layer where the attention weight is 
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
            """
            char_squish = F.tanh(self.attn_linear_w_1(output_char))
            char_attn = self.attn_linear_w_2(char_squish)
            char_attn.data.masked_fill_(x_mask.data.unsqueeze(2), -1e12)
            char_attn_norm = F.softmax(char_attn.squeeze(2))
            output = torch.bmm(output_char.transpose(1, 2), char_attn_norm.unsqueeze(2)).squeeze(2)
        else:
            """
            Mean pooling
            """
            x_lens = x_mask.data.eq(0).long().sum(dim=1)
            if self.use_cuda:
                weights = Variable(torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float())
            else:
                weights = Variable(torch.ones(x.size()) / x_lens.unsqueeze(1).float())
            weights.data.masked_fill_(x_mask.data, 0.0)
            output = torch.bmm(output_char.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        return output

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(2, batch_size, self.lstm_hidden)))
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(1, batch_size, self.lstm_hidden)))


class WordCharRNN(nn.Module):
    def __init__(self, n_classes, batch_size, num_tokens, embed_size, input_size, lstm_hidden, dropout=0.0,
                 attention=True, bidirectional=True, use_cuda=False, use_xavier_init_lstm=False):
        super(WordCharRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.attention = attention
        self.use_cuda = use_cuda

        self.drop = nn.Dropout(dropout)
        self.lookup = nn.Embedding(num_tokens, embed_size, padding_idx=0)

        b = 2 if self.bidirectional else 1

        self.word_lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True, dropout=dropout,
                                 bidirectional=self.bidirectional)
        if use_xavier_init_lstm:
            for weights in self.word_lstm.all_weights:
                for weight in weights:
                    if len(weight.data.shape) == 2:
                        nn.init.xavier_uniform(weight, gain=math.sqrt(2.0))

        if attention:
            self.attn_linear_w_1 = nn.Linear(b * lstm_hidden, b * lstm_hidden, bias=True)
            self.attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)
        self.linear = nn.Linear(b * lstm_hidden, n_classes)

    def forward(self, x, x_mask, c_emb, state_word):
        """
        x      : batch_size * length
        x_mask : batch_size * length
        x_c    : batch_size * length * emb_size
        """
        x_emb = self.lookup(x)
        cat_embed = torch.cat((x_emb, c_emb), 2)
        cat_embed = self.drop(cat_embed)
        output_word, state_word = self.word_lstm(cat_embed, state_word)
        output_word = self.drop(output_word)
        if self.attention:
            """
            An attention layer where the attention weight is 
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
            """
            word_squish = F.tanh(self.attn_linear_w_1(output_word))
            word_attn = self.attn_linear_w_2(word_squish)
            word_attn.data.masked_fill_(x_mask.data.unsqueeze(2), -1e12)
            word_attn_norm = F.softmax(word_attn.squeeze(2))
            word_attn_vectors = torch.bmm(output_word.transpose(1, 2), word_attn_norm.unsqueeze(2)).squeeze(2)
            output = self.linear(word_attn_vectors)
        else:
            """
            Mean pooling
            """
            x_lens = x_mask.data.eq(0).long().sum(dim=1)
            if self.use_cuda:
                weights = Variable(torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float())
            else:
                weights = Variable(torch.ones(x.size()) / x_lens.unsqueeze(1).float())
            weights.data.masked_fill_(x_mask.data, 0.0)
            word_vectors = torch.bmm(output_word.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
            output = self.linear(word_vectors)
        return output

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(2, batch_size, self.lstm_hidden)))
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(1, batch_size, self.lstm_hidden)))


class CombinedRNN(nn.Module):
    def __init__(self, n_classes, batch_size, num_word_tokens, embed_size,
                 candidate_input_size, lstm_hidden, dropout=0.0, attention=True, bidirectional=True, use_cuda=False,
                 use_xavier_init_lstm=False):
        super(CombinedRNN, self).__init__()

        self.batch_size = batch_size
        self.num_word_tokens = num_word_tokens
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.attention = attention
        self.use_cuda = use_cuda

        self.context_drop = nn.Dropout(dropout)
        self.candidate_drop = nn.Dropout(dropout)
        self.word_lookup = nn.Embedding(num_word_tokens, embed_size, padding_idx=0)

        b = 2 if self.bidirectional else 1

        # word_emb_dim + b * self.lstm_hidden_dim
        self.context_word_lstm = nn.LSTM(embed_size, lstm_hidden, batch_first=True, dropout=dropout,
                                         bidirectional=self.bidirectional)

        self.candidate_word_lstm = nn.LSTM(candidate_input_size, lstm_hidden, batch_first=True, dropout=dropout,
                                           bidirectional=self.bidirectional)

        if use_xavier_init_lstm:
            for weights in self.context_word_lstm.all_weights + \
                           self.candidate_word_lstm.all_weights:
                for weight in weights:
                    if len(weight.data.shape) == 2:
                        nn.init.xavier_uniform(weight, gain=math.sqrt(2.0))

        if attention:
            self.context_attn_linear_w_1 = nn.Linear(b * lstm_hidden, b * lstm_hidden, bias=True)
            self.context_attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)
            self.candidate_attn_linear_w_1 = nn.Linear(b * lstm_hidden, b * lstm_hidden, bias=True)
            self.candidate_attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)
        self.linear = nn.Linear(2 * b * lstm_hidden, n_classes)

        self.context_attention = []
        self.candidate_attention = []

    def forward(self,
                context_x_word, context_x_word_mask, state_context_word,
                candidate_x_word, candidate_x_word_mask, state_candidate_word,
                c_emb):
        """
        x      : batch_size * length
        x_mask : batch_size * length
        x_c    : batch_size * length * emb_size
        """
        context_x_emb = self.word_lookup(context_x_word)
        context_x_emb = self.context_drop(context_x_emb)
        output_context_word, state_context_word = self.context_word_lstm(context_x_emb, state_context_word)
        output_context_word = self.context_drop(output_context_word)

        x_emb = self.word_lookup(candidate_x_word)
        cat_embed = torch.cat((x_emb, c_emb), 2)
        cat_embed = self.candidate_drop(cat_embed)
        output_candidate_word, state_candidate_word = self.candidate_word_lstm(cat_embed, state_candidate_word)
        output_candidate_word = self.candidate_drop(output_candidate_word)

        if self.attention:
            context_attention_vectors, context_attention = self.attention_output(output_context_word,
                                                                                      context_x_word_mask,
                                                                                      self.context_attn_linear_w_1,
                                                                                      self.context_attn_linear_w_2)
            candidate_attention_vectors, candidate_attention = self.attention_output(output_candidate_word,
                                                                                          candidate_x_word_mask,
                                                                                          self.candidate_attn_linear_w_1,
                                                                                          self.candidate_attn_linear_w_2)
            if hasattr(self, 'context_attention'):
                self.context_attention.append(context_attention.numpy())
            else:
                self.context_attention = [context_attention.numpy()]
            if hasattr(self, 'candidate_attention'):
                self.candidate_attention.append(candidate_attention.numpy())
            else:
                self.candidate_attention = [candidate_attention.numpy()]
            output = self.linear(torch.cat((context_attention_vectors, candidate_attention_vectors), 1))
        else:
            context_vectors = self.mean_pooling_output(output_context_word, context_x_word, context_x_word_mask)
            candidate_vectors = self.mean_pooling_output(output_candidate_word, candidate_x_word, candidate_x_word_mask)
            output = self.linear(torch.cat((context_vectors, candidate_vectors)))
        return output

    def attention_output(self, output, x_mask, attn_linear_w_1, attn_linear_w_2):
        """
        An attention layer where the attention weight is
        a = T' . tanh(Wx + b)
        where x is the input, b is the bias.
        """
        word_squish = F.tanh(attn_linear_w_1(output))
        word_attn = attn_linear_w_2(word_squish)
        word_attn.data.masked_fill_(x_mask.data.unsqueeze(2), -1e12)
        word_attn_norm = F.softmax(word_attn.squeeze(2))
        word_attn_vectors = torch.bmm(output.transpose(1, 2), word_attn_norm.unsqueeze(2)).squeeze(2)
        return word_attn_vectors, word_attn.data

    def mean_pooling_output(self, output, x, x_mask):
        """
        Mean pooling
        """
        x_lens = x_mask.data.eq(0).long().sum(dim=1)
        if self.use_cuda:
            weights = Variable(torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float())
        else:
            weights = Variable(torch.ones(x.size()) / x_lens.unsqueeze(1).float())
        weights.data.masked_fill_(x_mask.data, 0.0)
        word_vectors = torch.bmm(output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        return word_vectors

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return ((Variable(torch.zeros(2, batch_size, self.lstm_hidden)),
                     Variable(torch.zeros(2, batch_size, self.lstm_hidden))),
                    (Variable(torch.zeros(2, batch_size, self.lstm_hidden)),
                     Variable(torch.zeros(2, batch_size, self.lstm_hidden))))
        else:
            return ((Variable(torch.zeros(1, batch_size, self.lstm_hidden)),
                     Variable(torch.zeros(1, batch_size, self.lstm_hidden))),
                    (Variable(torch.zeros(1, batch_size, self.lstm_hidden)),
                     Variable(torch.zeros(1, batch_size, self.lstm_hidden))))
