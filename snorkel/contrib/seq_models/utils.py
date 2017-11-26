import torch
import scipy
import re
import sys
import collections
from snorkel.models import Document, SequenceTag
import numpy as np


#
# Alignment Methods
#
def init_f(l, r, dim):
    while True:
        F = np.random.normal(0, 1.0, (l + r) * dim)
        f1 = F[:(l + r) * dim].reshape(((l + r), dim))
        product1 = np.dot(f1, f1.T)
        product1 = product1 - np.identity(product1.shape[0])
        if product1.any() == 0:
            continue
        return F

def magic_theta(z, theta):
    theta = torch.from_numpy(theta[:z.size(0) * z.size(1)]).view(z.size(0), z.size(1)).double()
    return torch.mul(z, torch.sign(torch.mul(z, theta)))

def procrustes(z, F):
    r = scipy.linalg.orthogonal_procrustes(z.numpy().T, F[:z.size(0) * z.size(1)].reshape(z.size(0), z.size(1)).T)[0]
    return torch.mm(torch.from_numpy(r), z)

#
# PCA
#
def get_singular_vectors(x, r=1, l=0, align_func=None):
    u, s, v = torch.svd(x)
    k = r if v.size(1) > l + r else v.size(1) - l
    z = torch.zeros(l + r, v.size(0)).double()
    z[0:l + k, ] = v.transpose(0, 1)[:l + k, ]
    # correct for alignment (e.g., procrustes, magic theta)
    z = align_func(z) if align_func else z
    return z[l:, ]

def get_principal_components(x, y=None, r=1, l=0, align_func=None):
    x = torch.from_numpy(x).float()
    mu = torch.mean(x, 0, keepdim=True)
    return get_singular_vectors(x - mu.repeat(x.shape[0], 1), r=r, align_func=align_func).numpy()

def init_f(l, r, dim):
    while True:
        F = np.random.normal(0, 1.0, (l + r) * dim)
        f1 = F[:(l + r) * dim].reshape(((l + r), dim))
        product1 = np.dot(f1, f1.T)
        product1 = product1 - np.identity(product1.shape[0])
        if product1.any() == 0:
            continue
        return F


def dist_exp_decay(length, decay, start, end):
    """
    Distance weight decay around span (start:end)

    :param length:
    :param decay:
    :param start:
    :param end:
    :return:
    """
    ret = []
    for i in range(length):
        if i < start:
            ret.append(decay ** (start - i))
        elif i >= end:
            ret.append(decay ** (i - end + 1))
        else:
            ret.append(1.)
    return np.array(ret)


def map_tag_seqs(tags, doc):
    """
    Take sequence tags, defined by absolute char offsets, and map to sentence/span objects
    :param:
    :param:
    :return tuple of sentence index and tag, (int, SequenceTag)
    """
    spans = []
    char_index = [s.abs_char_offsets[0] for s in doc.sentences]

    for t in tags:
        position = None
        for i in range(len(char_index) - 1):
            if t.abs_char_start >= char_index[i] and t.abs_char_end <= char_index[i+1]:
                position = i
                break
        if position == None and t.abs_char_start >= char_index[-1]:
            position = len(char_index) - 1
        if position == None:
            values = (doc.name, doc.id, t.abs_char_start, t.abs_char_end)
            sys.stderr.write("Warning! Skipping cross-sentence mention [{}] {} {}:{} \n".format(*values))
            continue
        try:
            shift = doc.sentences[position].abs_char_offsets[0]
            span = doc.sentences[position].text[t.abs_char_start-shift:t.abs_char_end-shift]
            spans.append((position, t, span))
        except Exception as e:
            print("Error!",e)

    return spans


def seq_tags_to_spans(seq_tags, documents, concept_type):
    # sort tags by document
    tags_by_doc = collections.defaultdict(list)
    for t in seq_tags:
        tags_by_doc[t.document_id].append(t)
    # sort docs by id
    docs_by_id = {doc.id:doc for doc in documents}
    # build spans (format is (SENTENCE POSITION, TAG, SPAN TEXT))
    spans = []
    for doc_id in tags_by_doc:
        spans.extend(map_tag_seqs(tags_by_doc[doc_id], docs_by_id[doc_id]))
    tags = [t[-1] for t in spans if t[1].concept_type == concept_type]
    return dict.fromkeys(tags)


def get_seq_tag_vocab(session, doc_ids, seq_tags, concept_type):
    docs = session.query(Document).filter(Document.id.in_(doc_ids)).all()
    return seq_tags_to_spans(seq_tags, docs, concept_type)


def seq_tags_to_doc_spans(seq_tags, documents, concept_types):

    # sort tags by document
    tags_by_doc = collections.defaultdict(list)
    for t in seq_tags:
        if t.concept_type not in concept_types:
            continue
        tags_by_doc[t.document_id].append(t)

    return {doc.id:map_tag_seqs(tags_by_doc[doc.id], doc) for doc in documents}


def char_to_word_index(char_offset, sentence):
    """
    Map absolute char offset to tokenized index

    :param char_offset:
    :param sentence:
    :return:
    """
    for i, co in enumerate(sentence.abs_char_offsets):
        if co <= char_offset < (co + len(sentence.words[i])):
            return i
    return i


def tag_sequence(tokens, fmt="IOB"):
    """

    :param tokens:
    :param fmt:
    :return:
    """
    assert len(tokens) > 0
    tags = ['O'] * len(tokens)
    if fmt == "IOB":
        tags[0] = 'B'
        tags[1:] = len(tags[1:]) * "I"
    elif fmt == "IOBES":
        if len(tags) == 1:
            tags[0] = 'S'
        else:
            tags[0] = 'B'
            tags[1:-1] = len(tags[1:-1]) * "I"
            tags[-1:] = "E"
    elif fmt == "IO":
        tags = ['I'] * len(tags)
    return tags


def get_tagged_sentences(session, doc_ids, concept_types, tag_fmt="IOB", verbose=False):
    """
    Return seq. tagged sentences

    :param doc_ids:
    :param concept_type:
    :return:
    """
    documents    = session.query(Document).filter(Document.id.in_(doc_ids)).all()
    #doc_uids     = [doc.id for doc in documents]
    seq_tags     = session.query(SequenceTag).filter(SequenceTag.document_id.in_(doc_ids)).all()
    tagged_spans = seq_tags_to_doc_spans(seq_tags, documents, concept_types)

    tagged = []
    for doc in documents:
        #index sequence labels by sentence
        label_index = collections.defaultdict(list)
        for position, seq_tag , mention in tagged_spans[doc.id]:
            label_index[position].append((seq_tag, mention))

        for s in doc.sentences:

            # TODO: force tokenization on special chars ?
            #for w in s.words:
            #    m = re.search("^([A-Z]{2,}[-][A-Z]{2,})$", w)

            tags = ['O'] * len(s.words)
            for seq_tag, mention in label_index[s.position]:
                i = char_to_word_index(seq_tag.abs_char_start, s)
                j = char_to_word_index(seq_tag.abs_char_end - 1, s)
                # sanity check
                s1 = ''.join(s.words[i:j + 1])
                s2 = mention.replace(" ","")
                if s1 != s2 and verbose:
                    sys.stderr.write("Sequence Alignment Error: {}\n".format(mention))
                    continue
                tags[i:j + 1] = ["{}-{}".format(t, seq_tag.concept_type) for t in tag_sequence(tags[i:j + 1], tag_fmt)]

            tagged.append((s.words, tags))

    return tagged


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % (method.__name__, te-ts))
        return result

    return timed