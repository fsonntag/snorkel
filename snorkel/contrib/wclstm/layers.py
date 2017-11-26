import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CharRNN(nn.Module):
    def __init__(self, batch_size, num_tokens, embed_size, lstm_hidden, dropout=0.0, attention=True, bidirectional=True, use_cuda=False):

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

        self.char_lstm = nn.LSTM(embed_size, lstm_hidden, batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        if attention:
            self.attn_linear_w_1 = nn.Linear(b * lstm_hidden, b * lstm_hidden, bias=True)
            self.attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)

    def forward(self, x, x_mask, state_char):
        """
        x      : batch_size * length
        x_mask : batch_size * length
        """
        x_emb = self.drop(self.lookup(x))
        output_char, state_char = self.char_lstm(x_emb, state_char)
        output_char = self.drop(output_char)
        if self.attention:
            """
            An attention layer where the attention weight is 
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
            """
            char_squish = F.tanh(self.attn_linear_w_1(output_char))
            char_attn = self.attn_linear_w_2(char_squish)
            char_attn.data.masked_fill_(x_mask.data,  -1e12)
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


class WordRNN(nn.Module):
    def __init__(self, n_classes, batch_size, num_tokens, embed_size, input_size, lstm_hidden, dropout=0.0, attention=True, bidirectional=True, use_cuda=False):
        super(WordRNN, self).__init__()

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

        self.word_lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
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
            word_attn.data.masked_fill_(x_mask.data, -1e12)
            word_attn_norm = F.softmax(word_attn.squeeze(2))
            word_attn_vectors = torch.bmm(output_word.transpose(1, 2), word_attn_norm.unsqueeze(2)).squeeze(2)
            ouptut = self.linear(word_attn_vectors)
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
            ouptut = self.linear(word_vectors)
        return ouptut

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(2, batch_size, self.lstm_hidden)))
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_hidden)),
                    Variable(torch.zeros(1, batch_size, self.lstm_hidden)))
