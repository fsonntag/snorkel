import numpy as np

from snorkel.contrib.rnn.rnn_base import RNNBase
from snorkel.contrib.rnn.utils import candidate_to_tokens




class TagRNN(RNNBase):

    OPEN, CLOSE = '~~[[~~', '~~]]~~'

    def __init__(self, save_file=None, name='TagRNN', seed=None, n_threads=4):
        """TagRNN for sequence tagging"""
        super(TagRNN, self).__init__(
            n_threads=n_threads, save_file=save_file, name=name, seed=seed
        )

    def _preprocess_data(self, candidates, extend):
        """Convert candidate sentences to tagged symbol sequences
            @candidates: candidates to process
            @extend: extend symbol table for tokens (train), or lookup (test)?
        """
        data, ends = [], []
        for candidate in candidates:
            # Read sentence data
            tokens = candidate_to_tokens(candidate)
            # Get label sequence
            labels = np.zeros(len(tokens), dtype=int)
            labels[candidate[0].get_word_start() : candidate[0].get_word_end()+1] = 1
            # Tag sequence
            s = self.tag(tokens, labels)
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(list(map(f, s))))
            ends.append(candidate[0].get_word_end())
        return data, ends

    def tag(self, seq, labels):
        assert (len(seq) == len(labels))
        seq_new, t = [], False
        for x, y in zip(seq, labels):
            if y and (not t):
                seq_new.append(self.OPEN)
                seq_new.append(x)
                t = True
            elif (not y) and t:
                seq_new.append(self.CLOSE)
                seq_new.append(x)
                t = False
            else:
                seq_new.append(x)
        return seq_new
