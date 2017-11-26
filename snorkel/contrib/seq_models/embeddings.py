import os
import sys
import logging
import numpy as np
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)

class EmbeddingLoader(object):
    """
    Simple wrapper class for loading various embedding file formats
    - FastText
    - Word2Vec (text, gensim)
    """
    def __init__(self, fpath, fmt="text", dim=None, verbose=True):
        """
        Load embeddings
        :param fpath:
        :param fmt:
        :param dim:
        """
        assert os.path.exists(fpath)

        self.fpath = fpath
        self.fmt = fmt
        self.dim = dim
        self.verbose = verbose

        # infer dimension
        if not self.dim and self.fmt != "gensim":
            header = open(self.fpath, "rU").readline().strip().split(' ')
            self.dim = len(header) - 1 if len(header) != 2 else int(header[-1])
            if self.verbose:
                logger.info("Detected {}d embeddings".format(self.dim))

    def _read(self):
        """
        Read embedding files from various formats.
        NOTE: Gensim models are ~2x faster loading than text

        :return:
        """
        errors = 0
        if self.fmt == "gensim":
            model = Word2Vec.load(self.fpath)
            model.init_sims()
            self.dim = model.wv.syn0norm.shape[1]
            if self.verbose:
                logger.info("Detected {}d embeddings".format(self.dim))

            for word in model.wv.vocab:
                i = model.wv.vocab[word].index
                yield (word, model.wv.syn0norm[i])

        elif self.fmt in ["text", "fasttext"]:
            start = 0 if self.fmt == "text" else 1
            for i, line in enumerate(open(self.fpath, "rU")):
                if i < start:
                    continue
                line = line.strip().split(' ')
                vec = np.array([float(x) for x in line[1:]])
                if len(vec) != self.dim:
                    errors += 1
                    continue
                yield (line[0], vec)

        if errors and self.verbose:
            logger.error("{} lines skipped\n".format(errors))

    def __iter__(self):
        return self._read()