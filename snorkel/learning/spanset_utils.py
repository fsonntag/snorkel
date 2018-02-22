import numpy as np

from snorkel.utils import overlapping_score


def merge_to_spansets_train(X, train_marginals, pred_marginals):
    candidate_spansets = ([], [])
    Y_pred = ([], [])

    for pred_i, marginals in enumerate([train_marginals, pred_marginals]):
        candidates = [(i, candidate) for i, candidate in enumerate(X)]
        candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start, c[1][0].char_end))
        current_spanset = []
        cardinality = marginals.shape[1]
        for value in range(cardinality - 1):
            for i, (original_i, candidate) in enumerate(candidates):
                if marginals[original_i].argmax() == value:
                    if current_spanset == []:
                        current_spanset.append((original_i, candidate))
                    else:
                        last_candidate = current_spanset[-1][1]
                        if overlapping_score(last_candidate[0], candidate[0]) > 0:
                            current_spanset.append((original_i, candidate))
                        else:
                            spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
                            for spanset_chunk in spanset_chunks:
                                pred_y = y_from_spanset_chunk(spanset_chunk, marginals, value)
                                candidate_spansets[pred_i].append(spanset_chunk)
                                Y_pred[pred_i].append(pred_y)
                            current_spanset = [(original_i, candidate)]

            spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
            for spanset_chunk in spanset_chunks:
                pred_y = y_from_spanset_chunk(spanset_chunk, marginals, value)
                candidate_spansets[pred_i].append(spanset_chunk)
                Y_pred[pred_i].append(pred_y)
            current_spanset = []

        for i, (original_i, candidate) in enumerate(candidates):
            if marginals[original_i].argmax() + 1 == cardinality:
                candidate_spansets[pred_i].append([(original_i, candidate)])
                Y_pred[pred_i].append(np.zeros(1, dtype=np.int))

    spansets_true, spansets_pred = candidate_spansets
    Y_true, Y_pred = Y_pred
    assert len(spansets_true) == len(Y_true)
    assert len(spansets_pred) == len(Y_pred)
    return spansets_true, Y_true, spansets_pred, Y_pred


def merge_to_spansets_dev(X, Y, marginals):
    candidate_spansets = []
    Y_true = []
    Y_pred = []

    candidates = [(i, candidate) for i, candidate in enumerate(X)]
    candidates.sort(key=lambda c: (c[1][0].sentence_id, c[1][0].char_start, c[1][0].char_end))
    current_spanset = []
    cardinality = marginals.shape[1]
    for value in range(cardinality - 1):
        for i, (original_i, candidate) in enumerate(candidates):
            if marginals[original_i].argmax() == value:
                if not current_spanset:
                    current_spanset.append((original_i, candidate))
                else:
                    last_candidate = current_spanset[-1][1]
                    if overlapping_score(last_candidate[0], candidate[0]) > 0:
                        current_spanset.append((original_i, candidate))
                    else:
                        spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
                        for spanset_chunk in spanset_chunks:
                            true_y, pred_y = ys_from_spanset_chunk(spanset_chunk, Y, marginals, value)
                            candidate_spansets.append(spanset_chunk)
                            Y_true.append(true_y)
                            Y_pred.append(pred_y)
                        current_spanset = [(original_i, candidate)]

        spanset_chunks = [current_spanset[x:x + 15] for x in range(0, len(current_spanset), 15)]
        for spanset_chunk in spanset_chunks:
            true_y, pred_y = ys_from_spanset_chunk(spanset_chunk, Y, marginals, value)
            candidate_spansets.append(spanset_chunk)
            Y_true.append(true_y)
            Y_pred.append(pred_y)
        current_spanset = []

    for i, (original_i, candidate) in enumerate(candidates):
        if marginals[original_i].argmax() + 1 == cardinality:
            candidate_spansets.append([(original_i, candidate)])
            Y_true.append(np.ravel(Y[Y.candidate_index[candidate.id]].todense()))
            Y_pred.append(np.zeros(1, dtype=np.int))

    assert len(candidate_spansets) == len(Y_pred)
    assert len(candidate_spansets) == len(Y_true)
    return candidate_spansets, Y_true, Y_pred


def y_from_spanset_chunk(spanset_chunk, marginals, value):
    spanset_marginals = np.asarray([marginals[s[0]] for s in spanset_chunk])
    pred_y = pred_from_spanset_marginals(spanset_marginals, spanset_chunk, value)
    return pred_y


def ys_from_spanset_chunk(spanset_chunk, Y, marginals, value):
    true_y = np.ravel([Y[Y.candidate_index[s[1].id]].todense() for s in spanset_chunk])
    spanset_marginals = np.asarray([marginals[s[0]] for s in spanset_chunk])
    pred_y = pred_from_spanset_marginals(spanset_marginals, spanset_chunk, value)
    return true_y, pred_y


def pred_from_spanset_marginals(spanset_marginals, spanset_chunk, value):
    marginal_column = spanset_marginals[:, value]
    best_candidate_row = None
    if len(marginal_column) == 2:
        two_max_indices = marginal_column.argsort()[-2:]
        two_max_values = marginal_column[two_max_indices]
        if abs(two_max_values[0] - two_max_values[1]) < 0.25:
            span1 = spanset_chunk[two_max_indices[0]][1][0]
            span2 = spanset_chunk[two_max_indices[1]][1][0]
            if span1.char_end - span1.char_start > span2.char_end - span2.char_start:
                best_candidate_row = two_max_indices[0]
            else:
                best_candidate_row = two_max_indices[1]
    elif len(marginal_column) == 3:
        three_max_indices = marginal_column.argsort()[-3:]
        three_max_values = marginal_column[three_max_indices]
        if abs(three_max_values[1] - three_max_values[2]) < 0.25:
            span1 = spanset_chunk[three_max_indices[1]][1][0]
            span2 = spanset_chunk[three_max_indices[2]][1][0]
            if 0 < (span1.char_end - span1.char_start) - (span2.char_end - span2.char_start):
                best_candidate_row = 1
            else:
                best_candidate_row = 2
            if abs(three_max_values[best_candidate_row] - three_max_values[0]) < 0.25:
                span1 = spanset_chunk[three_max_indices[0]][1][0]
                span2 = spanset_chunk[three_max_indices[best_candidate_row]][1][0]
                if 0 < (span1.char_end - span1.char_start) - (span2.char_end - span2.char_start):
                    best_candidate_row = three_max_indices[0]
                else:
                    best_candidate_row = three_max_indices[best_candidate_row]
            else:
                best_candidate_row = three_max_indices[best_candidate_row]
    elif len(marginal_column) > 3:
        four_max_indices = marginal_column.argsort()[-4:]
        four_max_values = marginal_column[four_max_indices]
        if abs(four_max_values[1] - four_max_values[2]) < 0.25:
            span1 = spanset_chunk[four_max_indices[2]][1][0]
            span2 = spanset_chunk[four_max_indices[3]][1][0]
            if 0 < (span1.char_end - span1.char_start) - (span2.char_end - span2.char_start):
                best_candidate_row = 2
            else:
                best_candidate_row = 3
            for i in range(1, -1, -1):
                if abs(four_max_values[best_candidate_row] - four_max_values[i]) < 0.25:
                    span1 = spanset_chunk[four_max_indices[i]][1][0]
                    span2 = spanset_chunk[four_max_indices[best_candidate_row]][1][0]
                    if 0 < (span1.char_end - span1.char_start) - (span2.char_end - span2.char_start):
                        best_candidate_row = i
                    else:
                        best_candidate_row = best_candidate_row
            best_candidate_row = four_max_indices[best_candidate_row]

    if best_candidate_row is None:
        best_candidate_row = np.argmax(spanset_marginals[:, value])
    pred_y = np.zeros(len(spanset_chunk), dtype=int)
    pred_y[best_candidate_row] = value + 1
    return pred_y
