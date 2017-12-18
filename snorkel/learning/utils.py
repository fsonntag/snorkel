import inspect
import os
import time
from collections import defaultdict
from itertools import product
from multiprocessing import Process, JoinableQueue

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

try:
    from queue import Empty
except:
    from Queue import Empty

from pandas import DataFrame


# matplotlib.use('Agg')
# warnings.filterwarnings("ignore", module="matplotlib")


############################################################
### General Learning Utilities
############################################################

def reshape_marginals(marginals):
    """Returns correctly shaped marginals as np array"""
    # Make sure training marginals are a numpy array first
    try:
        shape = marginals.shape
    except:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Set cardinality + marginals in proper format for binary v. categorical
    if len(shape) != 1:
        # If k = 2, make sure is M-dim array
        if shape[1] == 2:
            marginals = marginals[:, 1].reshape(-1)
    return marginals


class LabelBalancer(object):
    def __init__(self, y, categorical=False):
        """Utility class to rebalance training labels
        For example, to get the indices of a training set
        with labels y and around 90 percent negative examples,
            LabelBalancer(y).get_train_idxs(rebalance=0.1)
        """
        self.y = y
        if not categorical:
            self.y = np.ravel(self.y)

    def _get_pos(self, split):
        return np.where(self.y > (split + 1e-6))[0]

    def _get_neg(self, split):
        return np.where(self.y < (split - 1e-6))[0]

    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0 - pn)))
        s = int(round(float((1.0 - pn) * n) / pn))
        return (m, r) if r <= n else ((s, n) if s <= m else (m, n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0 - frac_pos)[::-1]

    def get_train_idxs(self, rebalance=False, split=0.5, rand_state=None):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
            @split: Split point for positive and negative classes
        """
        rs = np.random if rand_state is None else rand_state
        pos, neg = self._get_pos(split), self._get_neg(split)
        if rebalance:
            if len(pos) == 0:
                raise ValueError("No positive labels.")
            if len(neg) == 0:
                raise ValueError("No negative labels.")
            p = 0.5 if rebalance == True else rebalance
            n_neg, n_pos = self._get_counts(len(neg), len(pos), p)
            pos = rs.choice(pos, size=n_pos, replace=False)
            neg = rs.choice(neg, size=n_neg, replace=False)
        idxs = np.concatenate([pos, neg])
        rs.shuffle(idxs)
        return idxs

    def rebalance_categorical_train_idxs(self, rand_state=None):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
            @split: Split point for positive and negative classes
        """
        rs = np.random if rand_state is None else rand_state
        row_pos = []
        row_n = []
        cardinality = self.y.shape[1]
        max_indices = (self.y.argmax(axis=1) + 1) % cardinality
        for i in range(cardinality):
            curr_column_pos = np.where(max_indices == i)[0]
            if len(curr_column_pos) == 0:
                raise ValueError(f"No positive labels for row {i}.")
            row_pos.append(curr_column_pos)
            row_n.append(len(curr_column_pos))
        min_n = min(row_n)
        for i in range(len(row_pos)):
            row_pos[i] = row_pos[i][:min_n]
        idxs = np.concatenate(row_pos)
        rs.shuffle(idxs)
        return idxs

############################################################
### Advanced Scoring Classes
############################################################

class Scorer(object):
    """Abstract type for scorers"""

    def __init__(self, test_candidates, test_labels, gold_candidate_set=None):
        """
        :param test_candidates: A *list of Candidates* corresponding to
            test_labels
        :param test_labels: A *csrLabelMatrix* of ground truth labels for the
            test candidates
        :param gold_candidate_set: (optional) A *CandidateSet* containing the
            full set of gold labeled candidates
        """
        self.test_candidates = test_candidates
        self.test_labels = test_labels
        self.gold_candidate_set = gold_candidate_set

    def _get_cardinality(self, marginals):
        """Get the cardinality based on the marginals returned by the model."""
        if len(marginals.shape) == 1 or marginals.shape[1] < 3:
            cardinality = 2
        else:
            cardinality = marginals.shape[1]
        return cardinality

    def score(self, test_marginals, **kwargs):
        cardinality = self._get_cardinality(test_marginals)
        if cardinality == 2:
            return self._score_binary(test_marginals, **kwargs)
        else:
            return self._score_categorical(test_marginals, **kwargs)

    def _score_binary(self, test_marginals, train_marginals=None, b=0.5,
                      set_unlabeled_as_neg=True, display=True):
        raise NotImplementedError()

    def _score_categorical(self, test_marginals, train_marginals=None,
                           display=True):
        raise NotImplementedError()

    def summary_score(self, test_marginals, **kwargs):
        """Return the F1 score (for binary) or accuracy (for categorical)."""
        raise NotImplementedError()


class Counts:
    def __init__(self):
        self.tp, self.fp, self.tn, self.fn, self.fp_ov, self.fn_ov = \
            set(), set(), set(), set(), set(), set()
        self.types = set()
        self.t_tp, self.t_fp, self.t_tn, self.t_fn, self.t_fp_ov, self.t_fn_ov = \
            defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
        self.t_support = defaultdict(set)


class MentionScorer(Scorer):
    """Scorer for mention level assessment"""

    def _score_binary(self, test_marginals, train_marginals=None, b=0.5,
                      set_unlabeled_as_neg=True, set_at_thresh_as_neg=True, display=True,
                      **kwargs):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training
            candidates
        :param b: threshold for labeling
        :param set_unlabeled_as_neg: set test labels at the decision threshold
            of b as negative labels
        :param set_at_b_as_neg: set marginals at the decision threshold exactly
            as negative predictions
        :param display: show calibration plots?
        """
        test_label_array = []

        counts = Counts()

        candidates_by_types = defaultdict(list)
        for i, candidate in enumerate(self.test_candidates):
            candidates_by_types[candidate.__tablename__].append((i, candidate))

        for type, candidates in candidates_by_types.items():
            candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start))

            counts.types.add(type)
            for type_i, (i, candidate) in enumerate(candidates):
                test_label = self._get_label_for_candidate(i, candidate)

                # Set unlabeled examples to -1 by default
                if test_label == 0 and set_unlabeled_as_neg:
                    test_label = -1

                # Bucket the candidates for error analysis
                test_label_array.append(test_label)
                if test_label != 0:
                    if test_marginals[i] > b:
                        if test_label == 1:
                            counts.tp.add(candidate)
                            counts.t_tp[type].add(candidate)
                        else:
                            counts.fp.add(candidate)
                            counts.t_fp[type].add(candidate)
                            if self._overlapping_candidate_has_label({1}, type_i, candidate, candidates,
                                                                     set_unlabeled_as_neg):
                                counts.fp_ov.add(candidate)
                                counts.t_fp_ov[type].add(candidate)

                    elif test_marginals[i] < b or set_at_thresh_as_neg:
                        if test_label == -1:
                            counts.tn.add(candidate)
                            counts.t_tn[type].add(candidate)
                        else:
                            counts.fn.add(candidate)
                            counts.t_fn[type].add(candidate)
                            if self._overlapping_candidate_has_label({-1}, type_i, candidate, candidates,
                                                                     set_unlabeled_as_neg):
                                counts.fn_ov.add(candidate)
                                counts.t_fn_ov[type].add(candidate)

        # Calculate scores unadjusted for TPs not in our candidate set
        scores = scores_from_counts(counts, "Scores (Un-adjusted)", print_scores=display)
        if display:
            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_fn = [c for c in self.gold_candidate_set
                           if c not in self.test_candidates]
                print("\n")
                calculate_scores(len(counts.tp), len(counts.fp), len(counts.tn), len(counts.fn) + len(gold_fn),
                                 title="Corpus Recall-adjusted Scores")

            # If training and test marginals provided print calibration plots
            if train_marginals is not None and test_marginals is not None:
                print("\nCalibration plot:")
                calibration_plots(train_marginals, test_marginals,
                                  np.asarray(test_label_array))
        return scores

    def _get_label_for_candidate(self, i, candidate):
        # Handle either a LabelMatrix or else assume test_labels array is in
        # correct order i.e. same order as test_candidates
        try:
            test_label_index = self.test_labels.get_row_index(candidate)
            return self.test_labels[test_label_index, 0]
        except AttributeError:
            return self.test_labels[i]

    def _overlapping_candidate_has_label(self, labels, i, candidate, candidates, set_unlabeled_as_neg=True):
        j = i - 1
        before_overlapping = True
        while before_overlapping and 0 <= j:
            before_candidate = candidates[j][1]
            if before_candidate[0].sentence_id != candidate[0].sentence_id:
                break
            if candidate[0].char_start < before_candidate[0].char_end \
                    and candidate[0].char_end > before_candidate[0].char_start:
                test_label = self._get_label_for_candidate(j, before_candidate)
                if test_label == 0 and set_unlabeled_as_neg:
                    test_label = -1
                if test_label in labels:
                    return True
                j -= 1
            else:
                before_overlapping = False
        after_overlapping = True
        j = i + 1
        while after_overlapping and len(candidates) > j:
            after_candidate = candidates[j][1]
            if after_candidate[0].sentence_id != candidate[0].sentence_id:
                break
            if candidate[0].char_start < after_candidate[0].char_end \
                    and candidate[0].char_end > after_candidate[0].char_start:
                test_label = self._get_label_for_candidate(j, after_candidate)
                if test_label == 0 and set_unlabeled_as_neg:
                    test_label = -1
                if test_label in labels:
                    return True
                j += 1
            else:
                after_overlapping = False
        return False

    def _score_categorical(self, test_marginals, train_marginals=None,
                           display=True, **kwargs):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training
            candidates
        :param display: show calibration plots?
        """
        test_label_array = []
        correct = set()
        incorrect = set()
        counts = Counts()

        # Get predictions
        cardinality =  self._get_cardinality(test_marginals)
        test_pred = (test_marginals.argmax(axis=1) + 1) % cardinality

        candidates = [(i, candidate) for i, candidate in enumerate(self.test_candidates)]
        candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start))

        types = self.test_candidates[0].values[:-1]
        type_labels = set()
        for type in types:
            counts.types.add(type)
            type_labels.add(candidates[0][1].values.index(type) + 1)

        for type in types:
            type_label = candidates[0][1].values.index(type) + 1
            other_labels = type_labels - {type_label} | {0}

            for type_i, (i, candidate) in enumerate(candidates):
                test_label = self._get_label_for_candidate(i, candidate)
                if test_label != 0 and test_label == type_label:
                    counts.t_support[type].add(candidate)
                test_label_array.append(test_label)

                if test_pred[i] == type_label:
                    if test_label == type_label:
                        counts.tp.add(candidate)
                        counts.t_tp[type].add(candidate)
                    else:
                        counts.fp.add(candidate)
                        counts.t_fp[type].add(candidate)
                        if self._overlapping_candidate_has_label({test_pred[i]}, type_i, candidate, candidates, False):
                            counts.fp_ov.add(candidate)
                            counts.t_fp_ov[type].add(candidate)
                else:
                    if test_label != type_label:
                        counts.tn.add(candidate)
                        counts.t_tn[type].add(candidate)
                    else:
                        counts.fn.add(candidate)
                        counts.t_fn[type].add(candidate)
                        if self._overlapping_candidate_has_label(other_labels, type_i, candidate, candidates,
                                                                 False):
                            counts.fn_ov.add(candidate)
                            counts.t_fn_ov[type].add(candidate)

        # Calculate scores unadjusted for TPs not in our candidate set
        scores = scores_from_counts(counts, "Scores (Un-adjusted)", weighted=True, print_scores=display)
        if display:
            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_fn = [c for c in self.gold_candidate_set
                           if c not in self.test_candidates]
                print("\n")
                calculate_scores(len(counts.tp), len(counts.fp), len(counts.tn), len(counts.fn) + len(gold_fn),
                                 title="Corpus Recall-adjusted Scores")

            # If training and test marginals provided print calibration plots
            if train_marginals is not None and test_marginals is not None:
                print("\nCalibration plot:")
                calibration_plots(train_marginals, test_marginals,
                                  np.asarray(test_label_array))

        # Bucket the candidates for error analysis
        # for i, candidate in enumerate(self.test_candidates):
        #     # Handle either a LabelMatrix or else assume test_labels array is in
        #     # correct order i.e. same order as test_candidates
        #     try:
        #         test_label_index = self.test_labels.get_row_index(candidate)
        #         test_label = self.test_labels[test_label_index, 0]
        #     except AttributeError:
        #         test_label = self.test_labels[i]
        #     test_label_array.append(test_label)
        #     if test_label != 0:
        #         if test_pred[i] == test_label:
        #             correct.add(candidate)
        #         else:
        #             incorrect.add(candidate)
        # if display:
        #     nc, ni = len(correct), len(incorrect)
        #     print("Accuracy:", nc / float(nc + ni))
        #
        #     # If gold candidate set is provided calculate recall-adjusted scores
        #     if self.gold_candidate_set is not None:
        #         gold_missed = [c for c in self.gold_candidate_set
        #                        if c not in self.test_candidates]
        #         print("Coverage:", (nc + ni) / (nc + ni + len(gold_missed)))
        return scores

    def summary_score(self, test_marginals, **kwargs):
        """
        Return the F1 score (for binary) or accuracy (for categorical).
        Also return the label as second argument.
        """
        error_sets = self.score(test_marginals, display=False, **kwargs)
        if len(error_sets) == 1:
            _, _, f1 = binary_scores_from_counts(
                *map(len, [error_sets.tp, error_sets.fp, error_sets.fn, error_sets.fp]))
            return f1, "F1 Score"
        else:
            nc, ninc = map(len, error_sets)
            return nc / float(nc + ninc), "Accuracy"


def binary_scores_from_counts(ntp, nfp, ntn, nfn, nfp_ov=None, nfn_ov=None):
    """
    Precision, recall, and F1 scores from counts of TP, FP, TN, FN.
    Example usage:
        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0

    if not nfp_ov and not nfn_ov:
        return prec, rec, f1
    else:
        _fp_ov = nfp - nfp_ov
        _fn_ov = nfn - nfn_ov
        _tp_ov = ntp + nfp_ov + nfn_ov

        prec_ov = _tp_ov / (_tp_ov + _fp_ov) if _tp_ov + _fp_ov > 0 else 0.0
        rec_ov = _tp_ov / (_tp_ov + _fn_ov) if _tp_ov + _fn_ov > 0 else 0.0
        f1_ov = (2 * prec_ov * rec_ov) / (prec_ov + rec_ov) if prec_ov + rec_ov > 0 else 0.0

        _tp_half_ov = ntp + (nfp_ov + nfn_ov) / 2
        _fp_half_ov = nfp - nfp_ov
        _fn_half_ov = nfn - nfn_ov
        prec_half_ov = _tp_half_ov / (
                ntp + nfp_ov + nfn_ov + _fp_half_ov) if ntp + nfp_ov + nfn_ov + _fp_half_ov > 0 else 0.0
        rec_half_ov = _tp_half_ov / (
                ntp + nfp_ov + nfn_ov + _fn_half_ov) if ntp + nfp_ov + nfn_ov + _fn_half_ov > 0 else 0.0
        f1_half_ov = (2 * prec_half_ov * rec_half_ov) / (
                prec_half_ov + rec_half_ov) if prec_half_ov + rec_half_ov > 0 else 0.0
        return prec, rec, f1, prec_ov, rec_ov, f1_ov, prec_half_ov, rec_half_ov, f1_half_ov


def scores_from_counts(counts, title='Scores', weighted=False, print_scores=True):
    for type in counts.types:
        if print_scores:
            print(f"Scores for {type}")
        calculate_scores(len(counts.t_tp[type]),
                         len(counts.t_fp[type]),
                         len(counts.t_tn[type]),
                         len(counts.t_fn[type]),
                         len(counts.t_fp_ov[type]),
                         len(counts.t_fn_ov[type]),
                         title=title,
                         print_scores=print_scores)

    if weighted:
        weights = np.asarray([len(counts.t_support[type]) for type in counts.types])
        weight_sum = weights.sum()
        weights = weights / weight_sum

        if print_scores:
            print(f'General scores (weighted). Support: {"; ".join({f"{type}: {len(counts.t_support[type])}" for type in counts.types})}')
        scores = calculate_scores(np.average([len(counts.t_tp[type]) for type in counts.types], weights=weights),
                                  np.average([len(counts.t_fp[type]) for type in counts.types], weights=weights),
                                  np.average([len(counts.t_tn[type]) for type in counts.types], weights=weights),
                                  np.average([len(counts.t_fn[type]) for type in counts.types], weights=weights),
                                  np.average([len(counts.t_fp_ov[type]) for type in counts.types], weights=weights),
                                  np.average([len(counts.t_fn_ov[type]) for type in counts.types], weights=weights),
                                  title=title,
                                  print_scores=print_scores)
    else:
        if print_scores:
            print("General scores (micro)")
        scores = calculate_scores(len(counts.tp),
                                  len(counts.fp),
                                  len(counts.tn),
                                  len(counts.fn),
                                  len(counts.fp_ov),
                                  len(counts.fn_ov),
                                  title=title,
                                  print_scores=print_scores)
    return scores


def calculate_scores(ntp, nfp, ntn, nfn, nfp_ov=None, nfn_ov=None, title='Scores', print_scores=True):
    if nfp_ov or nfn_ov:
        return print_scores_with_overlapping(ntp, nfp, ntn, nfn, nfp_ov=nfp_ov, nfn_ov=nfn_ov, title='Scores',
                                             print_scores=print_scores)
    prec, rec, f1 = binary_scores_from_counts(ntp, nfp, ntn, nfn)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    if print_scores:
        print("========================================")
        print(title)
        print("========================================")
        print("Pos. class accuracy: {:.3}".format(pos_acc))
        print("Neg. class accuracy: {:.3}".format(neg_acc))
        print("Precision            {:.3}".format(prec))
        print("Recall               {:.3}".format(rec))
        print("F1                   {:.3}".format(f1))
        print("----------------------------------------")
        print("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
        print("========================================\n")
    return prec, rec, f1


def print_scores_with_overlapping(ntp, nfp, ntn, nfn, nfp_ov, nfn_ov, title='Scores', print_scores=True):
    prec, rec, f1, prec_ov, rec_ov, f1_ov, prec_half_ov, rec_half_ov, f1_half_ov \
        = binary_scores_from_counts(ntp, nfp, ntn, nfn, nfp_ov, nfn_ov)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    if print_scores:
        print("======================================================")
        print(title)
        print("======================================================")
        print("Pos. class accuracy:                            {:.3}".format(pos_acc))
        print("Neg. class accuracy:                            {:.3}".format(neg_acc))
        print("Precision                                       {:.3}".format(prec))
        print("Recall                                          {:.3}".format(rec))
        print("F1                                              {:.3}".format(f1))
        print("Overl. Precision                                {:.3}".format(prec_ov))
        print("Overl. Recall                                   {:.3}".format(rec_ov))
        print("Overl. F1                                       {:.3}".format(f1_ov))
        print("Half overl. Precision                           {:.3}".format(prec_half_ov))
        print("Half overl. Recall                              {:.3}".format(rec_half_ov))
        print("Half overl. F1                                  {:.3}".format(f1_half_ov))
        print("-------------------------------------------------------")
        print("TP: {} | FP: {} | TN: {} | FN: {} | FP OV: {} | FN OV: {}"
              .format(ntp, nfp, ntn, nfn, nfp_ov, nfn_ov))
        print("=======================================================\n")
    return prec, rec, f1, prec_ov, rec_ov, f1_ov, prec_half_ov, rec_half_ov, f1_half_ov


############################################################
### Calibration plots (currently unused, but should put back in?)
############################################################

def plot_prediction_probability(probs):
    plt.hist(probs, bins=20, normed=False, facecolor='blue')
    plt.xlim((0, 1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")


def plot_accuracy(probs, ground_truth):
    x = 0.1 * np.array(range(11))
    bin_assign = [x[i] for i in np.digitize(probs, x) - 1]
    correct = ((2 * (probs >= 0.5) - 1) == ground_truth)
    correct_prob = np.array([np.mean(correct[bin_assign == p]) for p in x])
    xc = x[np.isfinite(correct_prob)]
    correct_prob = correct_prob[np.isfinite(correct_prob)]
    plt.plot(x, np.abs(x - 0.5) + 0.5, 'b--', xc, correct_prob, 'ro-')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")


def calibration_plots(train_marginals, test_marginals, gold_labels=None):
    """Show classification accuracy and probability histogram plots"""
    n_plots = 3 if gold_labels is not None else 1

    # Whole set histogram
    plt.subplot(1, n_plots, 1)
    plot_prediction_probability(train_marginals)
    plt.title("(a) # Predictions (training set)")

    if gold_labels is not None:
        # Hold-out histogram
        plt.subplot(1, n_plots, 2)
        plot_prediction_probability(test_marginals)
        plt.title("(b) # Predictions (test set)")

        # Classification bucket accuracy
        plt.subplot(1, n_plots, 3)
        plot_accuracy(test_marginals, gold_labels)
        plt.title("(c) Accuracy (test set)")
    plt.show()


############################################################
### Grid search
############################################################

class GridSearch(object):
    """
    A class for running a hyperparameter grid search.

    :param model_class: The model class being trained
    :param parameter_dict: A dictionary of (hyperparameter name, list of values)
        pairs. Note that the hyperparameter name must correspond to a keyword
        argument in the `model_class.train` method.
    :param X_train: The training datapoints
    :param Y_train: If applicable, the training labels / marginals
    :param model_class_params: Keyword arguments to pass into model_class
        construction. Note that a new model is constructed for each new 
        combination of hyperparameters.
    :param model_hyperparams: Hyperparameters for the model- all must be
            keyword arguments to the `model_class.train` method. Any that are
            included in the grid search will be overwritten.
    :param save_dir: Note that checkpoints will be saved in save_dir/grid_search
    """

    def __init__(self, model_class, parameter_dict, X_train, Y_train=None,
                 model_class_params={}, model_hyperparams={}, save_dir='checkpoints'):
        self.model_class = model_class
        self.parameter_dict = parameter_dict
        self.param_names = parameter_dict.keys()
        self.X_train = X_train
        self.Y_train = Y_train
        self.model_class_params = model_class_params
        self.model_hyperparams = model_hyperparams
        self.save_dir = os.path.join(save_dir, 'grid_search')

    def search_space(self):
        return product(*[self.parameter_dict[pn] for pn in self.param_names])

    def fit(self, X_valid, Y_valid, b=0.5, beta=1, set_unlabeled_as_neg=True,
            n_threads=1, eval_batch_size=None):
        """
        Runs grid search, constructing a new instance of model_class for each
        hyperparameter combination, training on (self.X_train, self.Y_train),
        and validating on (X_valid, Y_valid). Selects the best model according
        to F1 score (binary) or accuracy (categorical).

        :param b: Scoring decision threshold (binary)
        :param beta: F_beta score to select model by (binary)
        :param set_unlabeled_as_neg: Set labels = 0 -> -1 (binary)
        :param n_threads: Parallelism to use for the grid search
        :param eval_batch_size: The batch_size for model evaluation
        """
        if n_threads > 1:
            opt_model, run_stats = self._fit_mt(X_valid, Y_valid, b=b,
                                                beta=beta, set_unlabeled_as_neg=set_unlabeled_as_neg,
                                                n_threads=n_threads, eval_batch_size=eval_batch_size)
        else:
            opt_model, run_stats = self._fit_st(X_valid, Y_valid, b=b,
                                                beta=beta, set_unlabeled_as_neg=set_unlabeled_as_neg,
                                                eval_batch_size=eval_batch_size)
        return opt_model, run_stats

    def _fit_st(self, X_valid, Y_valid, b=0.5, beta=1,
                set_unlabeled_as_neg=True, eval_batch_size=None):
        """Single-threaded implementation of `GridSearch.fit`."""
        # Iterate over the param values
        run_stats = []
        run_score_opt = -1.0
        for k, param_vals in enumerate(self.search_space()):
            start_ts = time.time()
            hps = self.model_hyperparams.copy()

            # Initiate the model from scratch each time
            # Some models may have seed set in the init procedure
            model = self.model_class(**self.model_class_params)
            model_name = '{0}_{1}'.format(model.name, k)

            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                hps[pn] = pv
            print("=" * 60)
            NUMTYPES = [float, int, np.float64]
            print("[%d] Testing %s" % (k + 1, ', '.join([
                "%s = %s" % (pn, ("%0.2e" % pv) if type(pv) in NUMTYPES else pv)
                for pn, pv in zip(self.param_names, param_vals)
            ])))
            print("=" * 60)

            # Train the model
            train_args = [self.X_train]
            if self.Y_train is not None:
                train_args.append(self.Y_train)

            # Pass in the dev set to the train method if applicable, for dev set
            # score printing, best-score checkpointing
            # Note: Need to set the save directory since passing in
            # (X_dev, Y_dev) will by default trigger checkpoint saving
            MAX_TRY = 10
            for idx in range(MAX_TRY):
                try:
                    if X_valid is not None:
                        model.train(*train_args, X_dev=X_valid, Y_dev=Y_valid,
                                    save_dir=self.save_dir, **hps)
                    else:
                        model.train(*train_args, **hps)
                    break
                except Exception as e:
                    print(e)

            # Test the model
            run_scores = model.score(X_valid, Y_valid, b=b, beta=beta,
                                     set_unlabeled_as_neg=set_unlabeled_as_neg,
                                     batch_size=eval_batch_size)

            if model.cardinality > 2:
                run_score, run_score_label = run_scores, "Accuracy"
                run_scores = [run_score]
            else:
                run_score = run_scores[-1]
                run_score_label = "F-{0} Score".format(beta)

            # Add scores to running stats, print, and set as optimal if best
            print("[{0}] {1}: {2}".format(model.name, run_score_label, run_score))
            run_stats.append(list(param_vals) + list(run_scores))
            if run_score > run_score_opt or k == 0:
                model.save(model_name=model_name, save_dir=self.save_dir)
                opt_model_name = model_name
                run_score_opt = run_score

            end_ts = time.time()
            print('GridSearch Iter: %2.2f sec' % (end_ts - start_ts))

        # Set optimal parameter in the learner model
        opt_model = self.model_class(**self.model_class_params)
        opt_model.load(opt_model_name, save_dir=self.save_dir)

        # Return optimal model & DataFrame of scores
        f_score = 'F-{0}'.format(beta)
        run_score_labels = ['Acc.'] if opt_model.cardinality > 2 else \
            ['Prec.', 'Rec.', f_score]
        sort_by = 'Acc.' if opt_model.cardinality > 2 else f_score
        self.results = DataFrame.from_records(
            run_stats, columns=self.param_names + run_score_labels
        ).sort_values(by=sort_by, ascending=False)
        return opt_model, self.results

    def _fit_mt(self, X_valid, Y_valid, b=0.5, beta=1,
                set_unlabeled_as_neg=True, n_threads=2, eval_batch_size=None):
        """Multi-threaded implementation of `GridSearch.fit`."""
        # First do a preprocessing pass over the data to make sure it is all
        # non-lazily loaded
        # TODO: Better way to go about it than this!!
        print("Loading data...")
        model = self.model_class(**self.model_class_params)
        _ = model._preprocess_data(self.X_train)
        _ = model._preprocess_data(X_valid)

        # Create queue of hyperparameters to test
        print("Launching jobs...")
        params_queue = JoinableQueue()
        param_val_sets = []
        for k, param_vals in enumerate(self.search_space()):
            param_val_sets.append(param_vals)
            hps = self.model_hyperparams.copy()
            for pn, pv in zip(self.param_names, param_vals):
                hps[pn] = pv
            params_queue.put((k, hps))

        # Create a queue to store output results
        scores_queue = JoinableQueue()

        # Start UDF Processes
        ps = []
        for i in range(n_threads):
            p = ModelTester(self.model_class, self.model_class_params,
                            params_queue, scores_queue, self.X_train, X_valid, Y_valid,
                            Y_train=self.Y_train, b=b, save_dir=self.save_dir,
                            set_unlabeled_as_neg=set_unlabeled_as_neg,
                            eval_batch_size=eval_batch_size)
            p.start()
            ps.append(p)

        # Collect scores
        run_stats = []
        while any([p.is_alive() for p in ps]):
            while True:
                try:
                    scores = scores_queue.get(True, QUEUE_TIMEOUT)
                    k = scores[0]
                    param_vals = param_val_sets[k]
                    run_stats.append([k] + list(param_vals) + list(scores[1:]))
                    print("Model {0} Done; score: {1}".format(k, scores[-1]))
                    scores_queue.task_done()
                except Empty:
                    break

        # Terminate the processes
        for p in ps:
            p.terminate()

        # Load best model; first element in each row of run_stats is the model
        # index, last one is the score to sort by
        # Note: the models may be returned out of order!
        i_opt = np.argmax([s[-1] for s in run_stats])
        k_opt = run_stats[i_opt][0]
        model = self.model_class(**self.model_class_params)
        model.load('{0}_{1}'.format(model.name, k_opt), save_dir=self.save_dir)

        # Return model and DataFrame of scores
        # Test for categorical vs. binary in hack-ey way for now...
        f_score = 'F-{0}'.format(beta)
        categorical = (len(scores) == 2)
        labels = ['Acc.'] if categorical else ['Prec.', 'Rec.', f_score]
        sort_by = 'Acc.' if categorical else f_score
        self.results = DataFrame.from_records(
            run_stats, columns=["Model"] + self.param_names + labels
        ).sort_values(by=sort_by, ascending=False)
        return model, self.results


QUEUE_TIMEOUT = 3


class ModelTester(Process):
    def __init__(self, model_class, model_class_params, params_queue,
                 scores_queue, X_train, X_valid, Y_valid, Y_train=None, b=0.5, beta=1,
                 set_unlabeled_as_neg=True, save_dir='checkpoints',
                 eval_batch_size=None):
        Process.__init__(self)
        self.model_class = model_class
        self.model_class_params = model_class_params
        self.params_queue = params_queue
        self.scores_queue = scores_queue
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.scorer_params = {
            'b': b,
            'beta': beta,
            'set_unlabeled_as_neg': set_unlabeled_as_neg,
            'batch_size': eval_batch_size
        }
        self.save_dir = save_dir

    def run(self):
        while True:
            # Get a new configuration from the queue
            try:
                k, hps = self.params_queue.get(True, QUEUE_TIMEOUT)

                # Initiate the model from scratch each time
                # Some models may have seed set in the init procedure
                model = self.model_class(**self.model_class_params)
                model_name = '{0}_{1}'.format(model.name, k)

                # Pass in the dev set to the train method if applicable, for dev 
                # set score printing, best-score checkpointing
                if 'X_dev' in inspect.getargspec(model.train):
                    hps['X_dev'] = self.X_valid
                    hps['Y_dev'] = self.Y_valid

                # Train model with given hyperparameters
                if self.Y_train is not None:
                    model.train(self.X_train, self.Y_train, **hps)
                else:
                    model.train(self.X_train, **hps)

                # Save the model
                # NOTE: Currently, we have to save every model because we are
                # testing asynchronously. This is obviously memory inefficient,
                # although probably not that much of a problem in practice...
                model.save(model_name=model_name, save_dir=self.save_dir)

                # Test the model
                run_scores = model.score(self.X_valid, self.Y_valid,
                                         **self.scorer_params)
                run_scores = [run_scores] if model.cardinality > 2 else \
                    list(run_scores)

                # Append score to out queue
                self.scores_queue.put([k] + run_scores, True, QUEUE_TIMEOUT)
            except Empty:
                break


class RandomSearch(GridSearch):
    """
    A GridSearch over a random subsample of the hyperparameter search space.

    :param seed: A seed for the GridSearch instance
    """

    def __init__(self, model_class, parameter_dict, X_train, Y_train=None, n=10,
                 model_class_params={}, model_hyperparams={}, seed=123,
                 save_dir='checkpoints', manual_param_grid=None):
        """Search a random sample of size n from a parameter grid"""
        self.rand_state = np.random.RandomState()
        self.rand_state.seed(seed)
        self.n = n
        self.seed = seed
        self.manual_param_grid = manual_param_grid
        super(RandomSearch, self).__init__(model_class, parameter_dict, X_train,
                                           Y_train=Y_train, model_class_params=model_class_params,
                                           model_hyperparams=model_hyperparams, save_dir=save_dir)

    def search_space(self):

        # use manually enumerated parameter grid
        if self.manual_param_grid:
            pass
            ##self.parameter_dict     = parameter_dict
            # self.param_names        = parameter_dict.keys()
            self.param_names = self.manual_param_grid['param_names']
            self.n = len(self.manual_param_grid["params"])
            return self.manual_param_grid["params"]

        else:
            self.rand_state.seed(self.seed)
            # fetch entire search space, shuffle it and return self.n param sets
            # we do this so that param sets are always proper subsets of larger self.n values
            params = list(super(RandomSearch, self).search_space())
            self.rand_state.shuffle(params)
            return params[:self.n]


############################################################
### Utility functions for annotation matrices
############################################################

def sparse_abs(X):
    """Element-wise absolute value of sparse matrix- avoids casting to dense matrix!"""
    X_abs = X.copy()
    if not sparse.issparse(X):
        return abs(X_abs)
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_abs.data = np.abs(X_abs.data)
    elif sparse.isspmatrix_lil(X):
        X_abs.data = np.array([np.abs(L) for L in X_abs.data])
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_abs


def candidate_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 0 (non-zero) labels.**
    """
    return np.where(sparse_abs(L).sum(axis=1) != 0, 1, 0).sum() / float(L.shape[0])


def LF_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    return np.ravel(sparse_abs(L).sum(axis=0) / float(L.shape[0]))


def candidate_overlap(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 1 (non-zero) labels.**
    """
    return np.where(sparse_abs(L).sum(axis=1) > 1, 1, 0).sum() / float(L.shape[0])


def LF_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) > 1, 1, 0).T * L_abs / float(L.shape[0]))


def candidate_conflict(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates which have > 1 (non-zero) labels _which are not equal_.**
    """
    return np.where(sparse_abs(L).sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).sum() / float(L.shape[0])


def LF_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).T * L_abs / float(L.shape[0]))


def LF_accuracies(L, labels):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate, and labels {-1,1}
    Return the accuracy of each LF w.r.t. these labels
    """
    return np.ravel(0.5 * (L.T.dot(labels) / sparse_abs(L).sum(axis=0) + 1))


def training_set_summary_stats(L, return_vals=True, verbose=False):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return simple summary statistics
    """
    N, M = L.shape
    coverage, overlap, conflict = candidate_coverage(L), candidate_overlap(L), candidate_conflict(L)
    if verbose:
        print("=" * 60)
        print("LF Summary Statistics: %s LFs applied to %s candidates" % (M, N))
        print("-" * 60)
        print("Coverage (candidates w/ > 0 labels):\t\t%0.2f%%" % (coverage * 100,))
        print("Overlap (candidates w/ > 1 labels):\t\t%0.2f%%" % (overlap * 100,))
        print("Conflict (candidates w/ conflicting labels):\t%0.2f%%" % (conflict * 100,))
        print("=" * 60)
    if return_vals:
        return coverage, overlap, conflict
