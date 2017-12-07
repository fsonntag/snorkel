from unittest import TestCase

from snorkel.annotations import csr_AnnotationMatrix
from snorkel.learning import MentionScorer, np, binary_scores_from_counts
from snorkel.models import candidate_subclass, Span, Sentence, Document, GoldLabelKey


class TestMentionScorer(TestCase):

    @classmethod
    def setUpClass(cls):
        Class = candidate_subclass('Class', ['class1'])
        document = Document(id=0, name='doc1')
        text = '.... aaaa .... bbbb .... cccc .... dddd .... eeee .... ffff .... gggg .... hhhh .... jjjj'
        sentence = Sentence(id=0, document_id=0, position=0, text=text, document=document)
        sentence.char_offsets = [5, 15, 25, 26, 32, 36, 45, 55, 62, 65, 75, 84]

        exact_1 = Span(char_start=5, char_end=8, sentence=sentence)
        c_exact_1 = Class(id=0, class1=exact_1)
        exact_2 = Span(char_start=55, char_end=58, sentence=sentence)
        c_exact_2 = Class(id=1, class1=exact_2)
        exact_3 = Span(char_start=75, char_end=78, sentence=sentence)
        c_exact_3 = Class(id=2, class1=exact_3)

        overlap_1_1 = Span(char_start=25, char_end=28, sentence=sentence)
        c_overlap_1_1 = Class(id=3, class1=overlap_1_1)
        overlap_1_2 = Span(char_start=26, char_end=27, sentence=sentence)
        c_overlap_1_2 = Class(id=4, class1=overlap_1_2)

        overlap_2_1 = Span(char_start=32, char_end=37, sentence=sentence)
        c_overlap_2_1 = Class(id=5, class1=overlap_2_1)
        overlap_2_2 = Span(char_start=36, char_end=42, sentence=sentence)
        c_overlap_2_2 = Class(id=6, class1=overlap_2_2)

        overlap_3_1 = Span(char_start=65, char_end=68, sentence=sentence)
        c_overlap_3_1 = Class(id=7, class1=overlap_3_1)
        overlap_3_2 = Span(char_start=62, char_end=71, sentence=sentence)
        c_overlap_3_2 = Class(id=8, class1=overlap_3_2)

        missing_1 = Span(char_start=45, char_end=48, sentence=sentence)
        c_missing_1 = Class(id=9, class1=missing_1)
        missing_2 = Span(char_start=84, char_end=87, sentence=sentence)
        c_missing_2 = Class(id=10, class1=missing_2)

        spurios = Span(char_start=15, char_end=19, sentence=sentence)
        c_spurios = Class(id=11, class1=spurios)

        cls.candidates = [c_exact_1, c_exact_2, c_exact_3, c_overlap_1_1, c_overlap_1_2, c_overlap_2_1, c_overlap_2_2,
                          c_overlap_3_1, c_overlap_3_2, c_missing_1, c_missing_2, c_spurios]
        labels = np.asarray([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0])
        cls.annotations = csr_AnnotationMatrix(labels.reshape(12, 1),
                                               candidate_index={i: i for i in range(len(labels))},
                                               row_index={i: i for i in range(len(labels))},
                                               annotation_key_cls=GoldLabelKey,
                                               col_index={0: 1}, key_index={1: 0})
        cls.marginals = np.asarray([0.6, 0.6, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.4, 0.6])

    def test_exact_strictness(self):
        scorer = MentionScorer(self.candidates, self.annotations)
        counts = scorer._score_binary(self.marginals)

        self.assertEqual(len(counts.tp), 3)
        self.assertEqual(len(counts.fp), 4)
        self.assertEqual(len(counts.fn), 5)

        prec, rec, f1 = binary_scores_from_counts(len(counts.tp), len(counts.fp), len(counts.tn), len(counts.fn))

        self.assertEqual(prec, 3 / 7)
        self.assertEqual(rec, 3 / 8)
        self.assertEqual(f1, 2 * (3 / 7 * 3 / 8) / (3 / 7 + 3 / 8))

    def test_overlapping_strictness(self):
        scorer = MentionScorer(self.candidates, self.annotations)
        counts = scorer._score_binary(self.marginals)

        self.assertEqual(len(counts.tp), 3)  # the 3 exact matches
        self.assertEqual(len(counts.fp) - len(counts.fp_ov), 1)  # the 1 spurious
        self.assertEqual(len(counts.fn) - len(counts.fn_ov), 2)  # the 2 missing
        self.assertEqual(len(counts.fp_ov), 3)  # the 3 overlapping
        self.assertEqual(len(counts.fn_ov), 3)  # the 3 overlapping

        prec, rec, f1, prec_ov, rec_ov, f1_ov, prec_half_ov, rec_half_ov, f1_half_ov \
            = binary_scores_from_counts(len(counts.tp), len(counts.fp), len(counts.tn), len(counts.fn),
                                        len(counts.fp_ov), len(counts.fn_ov))

        self.assertEqual(prec_ov, 9 / 10)
        self.assertEqual(rec_ov, 9 / 11)
        self.assertAlmostEqual(f1_ov, 2 * (9 / 10 * 9 / 11) / (9 / 10 + 9 / 11), places=5)

    def test_half_overlapping_strictness(self):
        scorer = MentionScorer(self.candidates, self.annotations)
        counts = scorer._score_binary(self.marginals)

        self.assertEqual(len(counts.tp), 3)  # the 3 exact matches
        self.assertEqual(len(counts.fp) - len(counts.fp_ov), 1)  # the 1 spurious
        self.assertEqual(len(counts.fn) - len(counts.fn_ov), 2)  # the 2 missing
        self.assertEqual(len(counts.fp_ov), 3)  # the 3 overlapping
        self.assertEqual(len(counts.fn_ov), 3)  # the 3 overlapping

        prec, rec, f1, prec_ov, rec_ov, f1_ov, prec_half_ov, rec_half_ov, f1_half_ov \
            = binary_scores_from_counts(len(counts.tp), len(counts.fp), len(counts.tn), len(counts.fn),
                                        len(counts.fp_ov), len(counts.fn_ov))

        self.assertEqual(prec_half_ov, (3 + 6 / 2) / 10)
        self.assertEqual(rec_half_ov, (3 + 6 / 2) / 11)
        self.assertEqual(f1_half_ov,
                         2 * ((3 + 6 / 2) / 10 * (3 + 6 / 2) / 11) / ((3 + 6 / 2) / 10 + (3 + 6 / 2) / 11))
