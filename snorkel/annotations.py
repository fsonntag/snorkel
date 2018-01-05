from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import *
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from pandas import DataFrame, Series
from sqlalchemy.sql import bindparam, select

from .features import get_span_feats
from .models import (
    GoldLabel, GoldLabelKey, Label, LabelKey, Feature, FeatureKey, Candidate,
    Marginal, LabelStat
)
from .models.meta import new_sessionmaker
from .udf import UDF, UDFRunner
from .utils import (
    matrix_conflicts,
    matrix_coverage,
    matrix_overlaps,
    matrix_tp,
    matrix_fp,
    matrix_fn,
    matrix_tn, overlapping_score)


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """

    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.candidate_index = kwargs.pop('candidate_index', None)
        self.row_index = kwargs.pop('row_index', None)
        self.annotation_key_cls = kwargs.pop('annotation_key_cls', None)
        self.key_index = kwargs.pop('key_index', None)
        self.col_index = kwargs.pop('col_index', None)

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, session, i):
        """Return the Candidate object corresponding to row i"""
        return session.query(Candidate).filter(Candidate.id == self.row_index[i]).one()

    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, session, j):
        """Return the AnnotationKey object corresponding to column j"""
        return session.query(self.annotation_key_cls) \
            .filter(self.annotation_key_cls.id == self.col_index[j]).one()

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def _get_sliced_indexes(self, s, axis, index, inv_index):
        """
        Remaps the indexes between matrix rows/cols and candidates/keys.
        Note: This becomes a massive performance bottleneck if not implemented
        properly, so be careful of changing!
        """
        if isinstance(s, slice):
            # Check for empty slice
            if s.start is None and s.stop is None:
                return index, inv_index
            else:
                idxs = np.arange(self.shape[axis])[s]
        elif isinstance(s, int):
            idxs = np.array([s])
        else:  # s is an array of ints
            idxs = s
            # If s is the entire slice, skip the remapping step
            if np.array_equal(idxs, list(range(len(idxs)))):
                return index, inv_index

        index_new, inv_index_new = {}, {}
        for i_new, i in enumerate(idxs):
            k = index[i]
            index_new[i_new] = k
            inv_index_new[k] = i_new
        return index_new, inv_index_new

    def __getitem__(self, key):
        X = super(csr_AnnotationMatrix, self).__getitem__(key)

        # If X is an integer or float value, just return it
        element_type = type(X)
        if element_type in [int, float] or issubclass(element_type, np.integer) \
                or issubclass(element_type, np.float):
            return X
        # If X is a matrix, make sure it stays a csr_AnnotationMatrix
        elif not isinstance(X, csr_AnnotationMatrix):
            X = csr_AnnotationMatrix(X)
        # X must be a matrix, so update appropriate csr_AnnotationMatrix fields
        X.annotation_key_cls = self.annotation_key_cls
        row_slice, col_slice = self._unpack_index(key)
        X.row_index, X.candidate_index = self._get_sliced_indexes(
            row_slice, 0, self.row_index, self.candidate_index)
        X.col_index, X.key_index = self._get_sliced_indexes(
            col_slice, 1, self.col_index, self.key_index)
        return X

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()


try:
    class csr_LabelMatrix(csr_AnnotationMatrix):

        def lf_stats(self, session, labels=None, est_accs=None):
            """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
            lf_names = [self.get_key(session, j).name for j in range(self.shape[1])]
            label_values = session.query(Candidate).filter(Candidate.id == self.row_index[0]).one().values

            # Default LF stats
            col_names = ['j', 'Coverage', 'Overlaps', 'Conflicts']
            d = {
                'j': list(range(self.shape[1])),
                'Coverage': Series(data=matrix_coverage(self), index=lf_names),
                'Overlaps': Series(data=matrix_overlaps(self), index=lf_names),
                'Conflicts': Series(data=matrix_conflicts(self), index=lf_names)
            }
            if labels is not None:
                col_names.extend(['TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
                ls = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
                tp = matrix_tp(self, ls)
                fp = matrix_fp(self, ls)
                fn = matrix_fn(self, ls)
                tn = matrix_tn(self, ls)
                ac = (tp + tn) / (tp + tn + fp + fn)
                d['Empirical Acc.'] = Series(data=ac, index=lf_names)
                d['TP'] = Series(data=tp, index=lf_names)
                d['FP'] = Series(data=fp, index=lf_names)
                d['FN'] = Series(data=fn, index=lf_names)
                d['TN'] = Series(data=tn, index=lf_names)

                write_confusion_matrices(self, ls, session, lf_names, label_values)

                write_fp(self, ls, session, lf_names)
                write_fn(self, ls, session, lf_names)

                write_missed(self, ls, session)

            if est_accs is not None:
                col_names.append('Learned Acc.')
                d['Learned Acc.'] = est_accs
                d['Learned Acc.'].index = lf_names
            return DataFrame(data=d, index=lf_names)[col_names]

        def store_falses(self, session, labels, clean=True):
            if clean == True:
                session.query(LabelStat).delete()
            lf_names = [self.get_key(session, j).name for j in range(self.shape[1])]
            ls = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
            store_tp(self, ls, session, lf_names)
            store_fp(self, ls, session, lf_names)
            store_tn(self, ls, session, lf_names)
            store_fn(self, ls, session, lf_names)


# This is a hack for getting the documentation to build...
except:
    class csr_LabelMatrix(object):
        def lf_stats(self, session, labels=None, est_accs=None):
            return None


class Annotator(UDFRunner):
    """Abstract class for annotating candidates and persisting these annotations to DB"""

    def __init__(self, annotation_class, annotation_key_class, f_gen):
        self.annotation_class = annotation_class
        self.annotation_key_class = annotation_key_class
        super(Annotator, self).__init__(AnnotatorUDF,
                                        annotation_class=annotation_class,
                                        annotation_key_class=annotation_key_class,
                                        f_gen=f_gen)

    def apply(self, split=0, key_group=0, replace_key_set=True, cids_query=None,
              **kwargs):
        # If we are replacing the key set, make sure the reducer key id cache is cleared!
        if replace_key_set:
            self.reducer.key_cache = {}

        # Get the cids based on the split, and also the count
        SnorkelSession = new_sessionmaker()
        session = SnorkelSession()
        cids_query = cids_query or session.query(Candidate.id) \
            .filter(Candidate.split == split)

        # Note: In the current UDFRunner implementation, we load all these into memory and fill a
        # multiprocessing JoinableQueue with them before starting... so might as well load them here and pass in.
        # Also, if we try to pass in a query iterator instead, with AUTOCOMMIT on, we get a TXN error...
        cids = cids_query.all()
        cids_count = len(cids)

        # Run the Annotator
        super(Annotator, self).apply(cids, split=split, key_group=key_group,
                                     replace_key_set=replace_key_set, cids_query=cids_query,
                                     count=cids_count, **kwargs)

        # Load the matrix
        return self.load_matrix(session, split=split, cids_query=cids_query,
                                key_group=key_group)

    def clear(self, session, split=0, key_group=0, replace_key_set=True,
              cids_query=None, **kwargs):
        """
        Deletes the Annotations for the Candidates in the given split.
        If replace_key_set=True, deletes *all* Annotations (of this Annotation sub-class)
        and also deletes all AnnotationKeys (of this sub-class)
        """
        query = session.query(self.annotation_class)

        # If replace_key_set=False, then we just delete the annotations for candidates in our split
        if not replace_key_set:
            sub_query = cids_query or session.query(Candidate.id) \
                .filter(Candidate.split == split)
            sub_query = sub_query.subquery()
            query = query.filter(self.annotation_class.candidate_id.in_(sub_query))
        query.delete(synchronize_session='fetch')

        # If we are creating a new key set, delete all old annotation keys
        if replace_key_set:
            query = session.query(self.annotation_key_class)
            query = query.filter(self.annotation_key_class.group == key_group)
            query.delete(synchronize_session='fetch')

    def apply_existing(self, split=0, key_group=0, cids_query=None, **kwargs):
        """Alias for apply that emphasizes we are using an existing AnnotatorKey set."""
        return self.apply(split=split, key_group=key_group,
                          replace_key_set=False, cids_query=cids_query, **kwargs)

    def load_matrix(self, session, split=0, key_group=0, cids_query=None,
                    **kwargs):
        raise NotImplementedError()


class AnnotatorUDF(UDF):
    def __init__(self, annotation_class, annotation_key_class, f_gen, **kwargs):
        self.annotation_class = annotation_class
        self.annotation_key_class = annotation_key_class

        # AnnotatorUDF relies on a *generator function* which yields annotations
        # given a candidate input
        # NB: inspect.isgeneratorfunction is not sufficient to check if f_ger
        # is a generator (does not work with fns that wrap gen, e.g. partial)
        # So no check here at the moment...
        self.anno_generator = f_gen

        # For caching key ids during the reduce step
        self.key_cache = {}

        super(AnnotatorUDF, self).__init__(**kwargs)

    def apply(self, cid, **kwargs):
        """
        Applies a given function to a Candidate, yielding a set of Annotations as key_name, value pairs

        Note: Accepts a candidate _id_ as argument, because of issues with putting Candidate subclasses
        into Queues (can't pickle...)
        """
        seen = set()
        cid = cid[0]
        c = self.session.query(Candidate).filter(Candidate.id == cid).one()
        for key_name, value in self.anno_generator(c):

            # Note: Make sure no duplicates emitted here!
            if (cid, key_name) not in seen:
                seen.add((cid, key_name))
                yield cid, key_name, value

    def reduce(self, y, clear, key_group, replace_key_set, **kwargs):
        """
        Inserts Annotations into the database.
        For Annotations with unseen AnnotationKeys (in key_group, if not None), either adds these
        AnnotationKeys if create_new_keyset is True, else skips these Annotations.
        """
        cid, key_name, value = y

        # Prepares queries
        # Annoation updating only needs to be done if clear=False
        if not clear:
            anno_update_query = self.annotation_class.__table__.update()
            anno_update_query = anno_update_query.where(self.annotation_class.candidate_id == bindparam('cid'))
            anno_update_query = anno_update_query.where(self.annotation_class.key_id == bindparam('kid'))
            anno_update_query = anno_update_query.values(value=bindparam('value'))

        # We only need to insert AnnotationKeys if replace_key_set=True
        # Note that in current configuration, we never update AnnotationKeys!
        if replace_key_set:
            key_insert_query = self.annotation_key_class.__table__.insert()

        # If we are replacing the AnnotationKeys (replace_key_set=True), then we assume they will
        # all have been handled by *this* reduce thread, and hence be in the cache already
        # So we only need key select queries if replace_key_set=False
        else:
            key_select_query = select([self.annotation_key_class.id]) \
                .where(self.annotation_key_class.name == bindparam('name'))
            if key_group is not None:
                key_select_query = key_select_query.where(self.annotation_key_class.group == key_group)

        anno_insert_query = self.annotation_class.__table__.insert()

        # Check if the AnnotationKey already exists, and gets its id
        key_id = None
        if key_name in self.key_cache:
            key_id = self.key_cache[key_name]
        else:
            key_args = {'name': key_name, 'group': key_group} if key_group else {'name': key_name}

            # If we are replacing the AnnotationKeys (replace_key_set=True), then we assume they will
            # all have been handled by *this* reduce thread, and hence be in the cache already
            if not replace_key_set:
                key_id = self.session.execute(key_select_query, key_args).first()

            # Key not in cache but exists in DB; add to cache
            if key_id is not None:
                key_id = key_id[0]
                self.key_cache[key_name] = key_id

            # Key not in cache or DB; add to both if create_new_keyset = True
            elif replace_key_set:
                key_id = self.session.execute(key_insert_query, key_args).inserted_primary_key[0]
                self.key_cache[key_name] = key_id

        # If AnnotationKey does not exist and create_new_keyset = False, skip
        if key_id is not None:

            # Updates the Annotation, assuming one might already exist, if try_update = True
            if not clear:
                res = self.session.execute(anno_update_query, {'cid': cid, 'kid': key_id, 'value': value})

            # If Annotation does not exist, insert
            if (clear or res.rowcount == 0) and value != 0:
                self.session.execute(anno_insert_query, {'candidate_id': cid, 'key_id': key_id, 'value': value})


def load_matrix(matrix_class, annotation_key_class, annotation_class, session,
                split=0, cids_query=None, key_group=0, key_names=None, zero_one=False,
                load_as_array=False):
    """
    Returns the annotations corresponding to a split of candidates with N members
    and an AnnotationKey group with M distinct keys as an N x M CSR sparse matrix.
    """
    cid_query = cids_query or session.query(Candidate.id) \
        .filter(Candidate.split == split)
    cid_query = cid_query.order_by(Candidate.id)

    keys_query = session.query(annotation_key_class.id)
    keys_query = keys_query.filter(annotation_key_class.group == key_group)
    if key_names is not None:
        keys_query = keys_query.filter(annotation_key_class.name.in_(frozenset(key_names)))
    keys_query = keys_query.order_by(annotation_key_class.id)

    # First, we query to construct the row index map
    cid_to_row = {}
    row_to_cid = {}
    for cid, in cid_query.all():
        if cid not in cid_to_row:
            j = len(cid_to_row)

            # Create both mappings
            cid_to_row[cid] = j
            row_to_cid[j] = cid

    # Second, we query to construct the column index map
    kid_to_col = {}
    col_to_kid = {}
    for kid, in keys_query.all():
        if kid not in kid_to_col:
            j = len(kid_to_col)

            # Create both mappings
            kid_to_col[kid] = j
            col_to_kid[j] = kid

    # Create sparse matrix in COO format for incremental construction
    row = []
    columns = []
    data = []

    # Rely on the core for fast iteration
    annot_select_query = annotation_class.__table__.select()

    # Iteratively construct row index and output sparse matrix
    # Cycles through the entire table to load the data.
    # Perfornamce may slow down based on table size; however, negligible since
    # it takes 8min to go throuh 245M rows (pretty fast).
    for res in session.execute(annot_select_query):
        # NOTE: The order of return seems to be switched in Python 3???
        # Either way, make sure the order is set here explicitly!
        cid, kid, val = res.candidate_id, res.key_id, res.value

        if (cid in cid_to_row) and (kid in kid_to_col):

            # Optionally restricts val range to {0,1}, mapping -1 -> 0
            if zero_one:
                val = 1 if val == 1 else 0
            row.append(cid_to_row[cid])
            columns.append(kid_to_col[kid])
            data.append(int(val))

    X = sparse.coo_matrix((data, (row, columns)), shape=(len(cid_to_row), len(kid_to_col)))

    # Return as an AnnotationMatrix
    Xr = matrix_class(X, candidate_index=cid_to_row, row_index=row_to_cid,
                      annotation_key_cls=annotation_key_class, key_index=kid_to_col,
                      col_index=col_to_kid)
    return np.squeeze(Xr.toarray()) if load_as_array else Xr


def load_label_matrix(session, **kwargs):
    return load_matrix(csr_LabelMatrix, LabelKey, Label, session, **kwargs)


def load_feature_matrix(session, **kwargs):
    return load_matrix(csr_AnnotationMatrix, FeatureKey, Feature, session, **kwargs)


def load_gold_labels(session, annotator_name, **kwargs):
    return load_matrix(csr_LabelMatrix, GoldLabelKey, GoldLabel, session, key_names=[annotator_name], **kwargs)


class LabelAnnotator(Annotator):
    """Apply labeling functions to the candidates, generating Label annotations

    :param lfs: A _list_ of labeling functions (LFs)
    """

    def __init__(self, lfs=None, label_generator=None):
        if lfs is not None:
            labels = lambda c: [(lf.__name__, lf(c)) for lf in lfs]
        elif label_generator is not None:
            labels = lambda c: label_generator(c)
        else:
            raise ValueError("Must provide lfs or label_generator kwarg.")

        # Convert lfs to a generator function
        # In particular, catch verbose values and convert to integer ones
        def f_gen(c):
            for lf_key, label in labels(c):
                # Note: We assume if the LF output is an int, it is already
                # mapped correctly
                if type(label) == int:
                    yield lf_key, label
                # None is a protected LF output value corresponding to 0,
                # representing LF abstaining
                elif label is None:
                    yield lf_key, 0
                elif label in c.values:
                    if c.cardinality > 2:
                        yield lf_key, c.values.index(label) + 1
                    # Note: Would be nice to not special-case here, but for
                    # consistency we leave binary LF range as {-1,0,1}
                    else:
                        val = 1 if c.values.index(label) == 0 else -1
                        yield lf_key, val
                else:
                    raise ValueError("""
                        Unable to parse label with value %s
                        for candidate with values %s""" % (label, c.values))

        super(LabelAnnotator, self).__init__(Label, LabelKey, f_gen)

    def load_matrix(self, session, **kwargs):
        return load_label_matrix(session, **kwargs)


class FeatureAnnotator(Annotator):
    """Apply feature generators to the candidates, generating Feature annotations"""

    def __init__(self, f=get_span_feats):
        super(FeatureAnnotator, self).__init__(Feature, FeatureKey, f)

    def load_matrix(self, session, **kwargs):
        return load_feature_matrix(session, **kwargs)


def save_marginals(session, X, marginals, training=True):
    """Save marginal probabilities for a set of Candidates to db.

    :param X: Either an M x N csr_AnnotationMatrix-class matrix, where M
        is number of candidates, N number of LFs/features; OR a list of
        arbitrary objects with candidate ids accessible via a .id attrib
    :param marginals: A dense M x K matrix of marginal probabilities, where
        K is the cardinality of the candidates, OR a M-dim list/array if K=2.
    :param training: If True, these are training marginals / labels; else they
        are saved as end model predictions.

    Note: The marginals for k=0 are not stored, only for k = 1,...,K
    """
    # Make sure that we are working with a numpy array
    try:
        shape = marginals.shape
    except:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Handle binary input as M x 1-dim array; assume elements represent
    # poksitive (k=1) class values
    if len(shape) == 1:
        marginals = np.vstack([1 - marginals, marginals]).T

    # Only add values for classes k=1,...,K
    marginal_tuples = []
    for i in range(shape[0]):
        for k in range(1, shape[1] if len(shape) > 1 else 2):
            if marginals[i, k] > 0:
                marginal_tuples.append((i, k, marginals[i, k]))

    # NOTE: This will delete all existing marginals of type `training`
    session.query(Marginal).filter(Marginal.training == training). \
        delete(synchronize_session='fetch')

    # Prepare bulk INSERT query
    q = Marginal.__table__.insert()

    # Check whether X is an AnnotationMatrix or not
    anno_matrix = isinstance(X, csr_AnnotationMatrix)
    if not anno_matrix:
        X = list(X)

    # Prepare values
    insert_vals = []
    for i, k, p in marginal_tuples:
        cid = X.get_candidate(session, i).id if anno_matrix else X[i].id
        insert_vals.append({
            'candidate_id': cid,
            'training': training,
            'value': k,
            # We cast p in case its a numpy type, which psycopg2 does not handle
            'probability': float(p)
        })

    # Execute update
    session.execute(q, insert_vals)
    session.commit()
    print("Saved %s marginals" % len(marginals))


def load_marginals(session, X=None, split=0, cids_query=None, training=True, marginal_limit=None):
    """Load the marginal probs. for a given split of Candidates"""
    # For candidate ids subquery
    cids_query = cids_query or session.query(Candidate.id) \
        .filter(Candidate.split == split)
    # Ensure ordering by CID
    cids_query = cids_query.order_by(Candidate.id)
    cids_sub_query = cids_query.subquery('cids')
    try:
        cardinality = X.get_candidate(session, 0).cardinality
    except:
        cardinality = X[0].cardinality

    # Load marginal tuples from db
    marginal_tuples_query = session.query(Marginal.candidate_id, Marginal.value,
                                          Marginal.probability) \
        .filter(Marginal.candidate_id == cids_sub_query.c.id) \
        .filter(Marginal.training == training)

    marginal_tuples = marginal_tuples_query.all()
    if marginal_limit:
        marginal_tuples = sorted(marginal_tuples, key=lambda x: (x[0], x[1]))[:marginal_limit * (cardinality - 1)]

    # If an AnnotationMatrix or list of candidates X is provided, we make sure
    # that the returned marginals are collated with X.
    if X is not None:
        # For now, handle feature matrix vs. list of objects with try / except
        # Handle AnnotationMatrix
        try:
            cardinality = X.get_candidate(session, 0).cardinality
            marginals = np.zeros((int(len(marginal_tuples) / (cardinality - 1)), cardinality))
            cid_map = X.candidate_index

        # Handle list of Candidates
        except:
            cardinality = X[0].cardinality
            marginals = np.zeros((int(len(marginal_tuples) / (cardinality - 1)), cardinality))
            cid_map = dict([(x.id, i) for i, x in enumerate(X)])

    # Otherwise if X is not provided, we sort by candidate id, using the
    # cids_query from above
    else:
        cardinality = session.query(Candidate) \
            .get(marginal_tuples[0][0]).cardinality
        marginals = np.zeros((cids_query.count(), cardinality))
        cid_map = dict([(cid, i) for i, (cid,) in enumerate(cids_query.all())])

    # Assemble the marginals matrix according to the candidate index of X
    for cid, k, p in marginal_tuples:
        marginals[cid_map[cid], k] = p

    # Add first column if k > 2, else ravel
    if cardinality > 2:
        row_sums = marginals.sum(axis=1)
        for i in range(marginals.shape[0]):
            marginals[i, 0] = 1 - row_sums[i]
    else:
        marginals = np.ravel(marginals[:, 1])
    return marginals


def write_confusion_matrices(L, labels, session, lf_names, label_values, seperate_files=False):
    out_path = Path('lf_stats')
    out_path.mkdir(exist_ok=True)
    max_value = int(L.max())
    max_label_value = int(labels.max())
    label_values = [str(v) for v in label_values]
    cm_index = label_values + ['Abstention']
    cm_columns = label_values
    all_cms = []
    all_ov_cms = []
    for lf_idx in range(L.shape[1]):
        cm = pd.DataFrame(np.zeros((max_value + 1, max_label_value + 1), dtype=int),
                          index=cm_index, columns=cm_columns)
        ov_cm = pd.DataFrame(np.zeros((max_value + 1, max_label_value + 1), dtype=int),
                             index=cm_index, columns=cm_columns)
        for i in range(max_value + 1):
            index_idx = cm_index[(i - 1) % (max_value + 1)]
            for j in range(max_label_value + 1):
                column_idx = cm_columns[j - 1]
                pred_indices = sparse.find(L[:, lf_idx] == i)[0]
                true_indices = np.where(labels == j)[0]
                overlap_indices = set(pred_indices) & set(true_indices)
                cm[column_idx][index_idx] = len(overlap_indices)
                ov_cm[column_idx][index_idx] = len(overlap_indices)
                if i > 0 and i < max_value and j != i and len(overlap_indices) > 0:
                    true_indices = np.where(labels == i)[0]
                    pred_candidate_indices = [L.row_index[i_idx] for i_idx in overlap_indices]
                    pred_candidates = session.query(Candidate).filter(Candidate.id.in_(pred_candidate_indices)).all()
                    true_candidate_indices = [L.row_index[j_idx] for j_idx in true_indices]
                    true_candidates = session.query(Candidate).filter(Candidate.id.in_(true_candidate_indices)).all()
                    for pred_candidate in pred_candidates:
                        overlapping_candidates = [true_candidate for true_candidate in true_candidates
                                                  if 1.0 > overlapping_score(true_candidate, pred_candidate) > 0.5]

                        ov_cm[column_idx][index_idx] -= len(overlapping_candidates)

        all_cms.append(cm)
        all_ov_cms.append(ov_cm)
        if seperate_files:
            with open(out_path / f'cm_{lf_names[lf_idx]}.txt', 'w') as file:
                file.write(str(cm.T))
            with open(out_path / f'ov_cm_{lf_names[lf_idx]}.txt', 'w') as file:
                file.write(str(ov_cm.T))
    if not seperate_files:
        with open(out_path / f'cm_all.txt', 'w') as file:
            for i, (cm, ov_cm) in enumerate(zip(all_cms, all_ov_cms)):
                file.write(f'{lf_names[i]}\n{str(cm.T)}\n{str(ov_cm.T)}\n\n')


def write_fp(L, labels, session, lf_names):
    out_path = Path('lf_stats')
    out_path.mkdir(exist_ok=True)
    for i in range(L.shape[1]):
        candidates = get_fp_candidates(L, labels, i, session)
        if candidates:
            with open(out_path / f'fp_{lf_names[i]}.tsv', 'w') as file:
                file.writelines([f'{c[0].get_span()}\t{c[0].stable_id}\n' for c in candidates])
            candidate_names = [c[0].get_span() for c in candidates]
            name_counts = Counter(candidate_names)
            with open(out_path / f'fp_counts_{lf_names[i]}.tsv', 'w') as file:
                file.writelines([f'{name}\t{count}' + '\n' for name, count in name_counts.most_common()])


def write_fn(L, labels, session, lf_names):
    out_path = Path('lf_stats')
    out_path.mkdir(exist_ok=True)
    for i in range(L.shape[1]):
        candidates = get_fn_candidates(L, labels, i, session)
        if candidates:
            with open(out_path / f'fn_{lf_names[i]}.tsv', 'w') as file:
                file.writelines([f'{c[0].get_span()}\t{c[0].stable_id}\n' for c in candidates])
            candidate_names = [c[0].get_span() for c in candidates]
            name_counts = Counter(candidate_names)
            with open(out_path / f'fn_counts_{lf_names[i]}.tsv', 'w') as file:
                file.writelines([f'{name}\t{count}' + '\n' for name, count in name_counts.most_common()])


def store_tp(L, labels, session, lf_names):
    for i in range(L.shape[1]):
        candidates = get_tp_candidates(L, labels, i, session)
        candidate_names = [(c[0].get_span(), value) for c, value in candidates]
        name_counts = Counter(candidate_names)
        label_key = session.query(LabelKey).filter(LabelKey.name == lf_names[i]).one()
        for (name, value), count in name_counts.most_common():
            label_stat = LabelStat(true_label=value, key=label_key, span_value=name, frequency=count, type='tp')
            session.add(label_stat)
        session.commit()


def store_fp(L, labels, session, lf_names):
    for i in range(L.shape[1]):
        candidates = get_fp_candidates(L, labels, i, session)
        candidate_names = [(c[0].get_span(), value) for c, value in candidates]
        name_counts = Counter(candidate_names)
        label_key = session.query(LabelKey).filter(LabelKey.name == lf_names[i]).one()
        for (name, value), count in name_counts.most_common():
            label_stat = LabelStat(true_label=value, key=label_key, span_value=name, frequency=count, type='fp')
            session.add(label_stat)
        session.commit()


def store_tn(L, labels, session, lf_names):
    for i in range(L.shape[1]):
        candidates = get_tn_candidates(L, labels, i, session)
        candidate_names = [(c[0].get_span(), value) for c, value in candidates]
        name_counts = Counter(candidate_names)
        label_key = session.query(LabelKey).filter(LabelKey.name == lf_names[i]).one()
        for (name, value), count in name_counts.most_common():
            label_stat = LabelStat(true_label=value, key=label_key, span_value=name, frequency=count, type='tn')
            session.add(label_stat)
        session.commit()


def store_fn(L, labels, session, lf_names):
    for i in range(L.shape[1]):
        candidates = get_fn_candidates(L, labels, i, session)
        candidate_names = [(c[0].get_span(), value) for c, value in candidates]
        name_counts = Counter(candidate_names)
        label_key = session.query(LabelKey).filter(LabelKey.name == lf_names[i]).one()
        for (name, value), count in name_counts.most_common():
            label_stat = LabelStat(true_label=value, key=label_key, span_value=name, frequency=count, type='fn')
            session.add(label_stat)
        session.commit()


def write_missed(L, labels, session):
    out_path = Path('lf_stats')
    out_path.mkdir(exist_ok=True)
    max_value = int(labels.max())
    false_indices = [L.row_index[j] for j in np.where(labels == 0)[0]]
    candidate_indices = []
    for i in range(1, max_value + 1):
        candidate_indices += [L.row_index[j] for j in np.where(labels == i)[0]]
    false_candidates = set(session.query(Candidate).filter(Candidate.id.in_(false_indices)).all())
    true_candidates = set(session.query(Candidate).filter(Candidate.id.in_(candidate_indices)).all())
    true_candidate_names = {c[0].get_span() for c in true_candidates}

    with open(out_path / 'potential_missed_annotations', 'w') as file:
        for false_candidate in false_candidates:
            if false_candidate[0].get_span() in true_candidate_names:
                in_overlapping = False
                overlapping_candidates = [candidate for candidate in true_candidates
                                          if candidate[0].sentence_id == false_candidate[0].sentence_id
                                          and candidate[0].char_start < false_candidate[0].char_end
                                          and candidate[0].char_end > false_candidate[0].char_start]
                for ov_candidate in overlapping_candidates:
                    if ov_candidate in true_candidates:
                        in_overlapping = True
                        break
                if not in_overlapping:
                    file.write(f'{false_candidate[0].get_span()}\t{false_candidate[0].stable_id}\n')


def get_tp_candidates(L, labels, lf_idx, session):
    max_value = int(L.max())
    positives = (L[:, lf_idx] > 0).todense()
    positive_values = np.multiply(L[:, lf_idx].toarray(), positives)
    positives = positive_values < max_value
    positive_values = np.multiply(L[:, lf_idx].toarray(), positives)
    tp_indices = np.where((positive_values > 0) & (positive_values == labels.reshape(-1, 1)))[0]
    tp_candidate_ids = [L.row_index[tp_index] for tp_index in tp_indices]
    tp_candidates = session.query(Candidate).filter(Candidate.id.in_(tp_candidate_ids)).all()
    tp_candidates = [(tp_candidate, int(value))
                     for tp_candidate, value
                     in zip(tp_candidates, np.ravel(positive_values[positive_values > 0]))]
    return tp_candidates


def get_fp_candidates(L, labels, lf_idx, session):
    label_max_value = int(labels.max())
    all_fp_candidates = []
    for pos_value in range(1, label_max_value + 1):
        true_candidate_indices = [L.row_index[j] for j in np.where(labels == pos_value)[0]]
        true_candidates = set(session.query(Candidate).filter(Candidate.id.in_(true_candidate_indices)).all())

        positives = (L[:, lf_idx] == pos_value).todense()
        positive_values = np.multiply(L[:, lf_idx].toarray(), positives)
        fp_indices = np.where(((positive_values > 0) & (positive_values != labels.reshape(-1, 1))) == True)[0]
        fp_candidate_ids = [L.row_index[fp_index] for fp_index in fp_indices]
        fp_candidates = session.query(Candidate).filter(Candidate.id.in_(fp_candidate_ids)).all()
        fp_candidates = [(tp_candidate, int(value))
                         for tp_candidate, value
                         in zip(fp_candidates, np.ravel(positive_values[positive_values > 0]))]
        for fp_candidate in fp_candidates:
            overlapping_candidates = [true_candidate for true_candidate in true_candidates
                                      if overlapping_score(true_candidate, fp_candidate[0]) > 0.5]
            if not overlapping_candidates:
                all_fp_candidates.append(fp_candidate)

    return all_fp_candidates


def get_tn_candidates(L, labels, lf_idx, session):
    false_value = int(L.max())
    negatives = np.invert((L[:, lf_idx] == false_value).toarray()).astype(int)
    tn_indices = np.where((negatives == 0) & (negatives == labels.reshape(-1, 1)))[0]
    candidate_ids = [L.row_index[tp_index] for tp_index in tn_indices]
    candidates = session.query(Candidate).filter(Candidate.id.in_(candidate_ids)).all()
    candidates = [(candidate, 0) for candidate in candidates]
    return candidates


def get_fn_candidates(L, labels, lf_idx, session):
    false_value = int(L.max())
    negatives = np.invert((L[:, lf_idx] == false_value).toarray()).astype(int)
    fn_indices = np.where((negatives == 0) & (negatives != labels.reshape(-1, 1)))[0]
    candidate_ids = [L.row_index[fn_index] for fn_index in fn_indices]
    candidates = session.query(Candidate).filter(Candidate.id.in_(candidate_ids)).all()
    candidates = [(candidate, int(value)) for candidate, value in zip(candidates, labels[fn_indices])]
    return candidates
