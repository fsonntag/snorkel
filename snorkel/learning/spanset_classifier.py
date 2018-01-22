from itertools import chain

from snorkel.learning import MentionScorer
from snorkel.learning.classifier import Classifier
from snorkel.learning.spanset_utils import *


class SpansetClassifier(Classifier):

    def __init__(self, output_path=None, **kwargs):
        self.output_path = output_path
        super(SpansetClassifier, self).__init__(**kwargs)

    def spanset_error_analysis(self, session, X_test, X_test_transformed, Y_test,
                               gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True, display=True,
                               scorer=MentionScorer, **kwargs):
        """
        Prints full score analysis using the Scorer class, and then returns the
        a tuple of sets conatining the test candidates bucketed for error
        analysis, i.e.:
            * For binary: TP, FP, TN, FN
            * For categorical: correct, incorrect

        :param X_test: The input test candidates, as a list or annotation matrix
        :param Y_test: The input test labels, as a list or annotation matrix
        :param gold_candidate_set: Full set of TPs in the test set
        :param b: Decision boundary *for binary setting only*
        :param set_unlabeled_as_neg: Whether to map 0 labels -> -1, *binary setting*
        :param display: Print score report
        :param scorer: The Scorer sub-class to use
        """
        # Compute the marginals

        prediction_type = kwargs.get('prediction_type', False)

        test_marginals = self.marginals(X_test_transformed, **kwargs)

        if prediction_type == 'train':
            spansets_true, Y_true, spansets_pred, Y_pred = merge_to_spansets_train(X_test, Y_test, test_marginals)
            spansets_true, Y_true, spansets_pred, Y_pred = list(chain.from_iterable(spansets_true)), np.hstack(Y_true), \
                                                           list(chain.from_iterable(spansets_pred)), np.hstack(Y_pred)
            X_test1, Y_test = (list(t) for t in zip(*sorted(zip(spansets_true, Y_true),
                                                            key=lambda x: (x[0][1][0].sentence_id,
                                                                           x[0][1][0].char_start,
                                                                           x[0][1][0].char_end))))
            X_test2, Y_pred = (list(t) for t in zip(*sorted(zip(spansets_pred, Y_pred),
                                                            key=lambda x: (x[0][1][0].sentence_id,
                                                                           x[0][1][0].char_start,
                                                                           x[0][1][0].char_end))))
            assert X_test1 == X_test2
            X_test = [x[1] for x in X_test1]
        else:
            X_test, Y_test, Y_pred = merge_to_spansets_dev(X_test, Y_test, test_marginals)
            X_test, Y_test, Y_pred = list(chain.from_iterable(X_test)), np.hstack(Y_test), np.hstack(Y_pred)
            X_test = [x[1] for x in X_test]

        # Initialize and return scorer
        s = scorer(X_test, Y_test, gold_candidate_set=gold_candidate_set, output_path=self.output_path)
        return s._score_categorical(Y_pred, train_marginals=None, b=b,
                                    display=display, set_unlabeled_as_neg=set_unlabeled_as_neg,
                                    already_predicted=True, prediction_type=prediction_type)
