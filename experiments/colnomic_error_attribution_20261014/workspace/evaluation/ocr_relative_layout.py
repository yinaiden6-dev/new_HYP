"""Compare explicit pairwise relative OCR graphs; absolute positions are discarded."""
from dataclasses import dataclass

import numpy as np

from .ocr_layout import extract_lines, matched_pairs, reference_idf


@dataclass
class RelativeLines:
    tokens: list
    weights: np.ndarray
    postings: dict
    neighbors: np.ndarray
    relative_xy: np.ndarray
    relative_log_size: np.ndarray
    relative_direction: np.ndarray
    edge_weights: np.ndarray


def extract_relative_lines(record, idf, settings):
    lines = extract_lines(record, idf)
    n = len(lines.tokens)
    k = min(settings['relative_neighbors'], max(0, n - 1))
    if k == 0:
        return RelativeLines(lines.tokens, lines.weights, lines.postings,
                             np.empty((n, 0), dtype=np.int64), np.empty((n, 0, 2)),
                             np.empty((n, 0, 2)), np.empty((n, 0, 2)), np.empty((n, 0)))
    # i -> j in line i's reading frame, normalized by geometric-mean line height.
    delta = lines.centers[None, :, :] - lines.centers[:, None, :]
    normalization = np.sqrt(lines.heights[:, None] * lines.heights[None, :])
    direction = lines.directions[:, None, :]
    relative_xy = np.stack(((delta * direction).sum(2),
                            direction[:, :, 0] * delta[:, :, 1] -
                            direction[:, :, 1] * delta[:, :, 0]), axis=2) / normalization[:, :, None]
    distance = np.linalg.norm(relative_xy, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind='stable')[:, :k]
    row = np.arange(n)[:, None]
    other_direction = lines.directions[neighbors]
    relative_direction = np.stack(((direction * other_direction).sum(2),
                                   direction[:, :, 0] * other_direction[:, :, 1] -
                                   direction[:, :, 1] * other_direction[:, :, 0]), axis=2)
    log_size = np.stack((np.log(lines.widths[neighbors] / lines.widths[:, None]),
                         np.log(lines.heights[neighbors] / lines.heights[:, None])), axis=2)
    edge_weights = np.sqrt(lines.weights[:, None] * lines.weights[neighbors])
    # Centers, absolute dimensions and global directions never enter the matcher.
    return RelativeLines(lines.tokens, lines.weights, lines.postings, neighbors,
                         relative_xy[row, neighbors], log_size, relative_direction, edge_weights)


def relation_agreement(qxy, rxy, qsize, rsize, qdirection, rdirection, settings):
    position_error = ((qxy - rxy) / settings['relative_distance_sigma']) ** 2
    size_error = ((qsize - rsize) / settings['relative_log_size_sigma']) ** 2
    orientation = np.clip((qdirection * rdirection).sum(-1), 0., 1.) ** 2
    # Preserve left/right and above/below once both displacements clear tolerance.
    tolerance = settings['relative_order_tolerance']
    reversed_order = (((qxy * rxy) < 0) & (np.abs(qxy) > tolerance) &
                      (np.abs(rxy) > tolerance)).any(-1)
    return np.where(reversed_order, 0., orientation *
                    np.exp(-.5 * (position_error.sum(-1) + size_error.sum(-1))))


def _assignment(qi, ri, qualities, weights, settings):
    selected, used_q, used_r = [], set(), set()
    for i in np.argsort(-qualities * weights[qi], kind='stable'):
        q, r = int(qi[i]), int(ri[i])
        if qualities[i] >= settings['minimum_relation_quality'] and q not in used_q and r not in used_r:
            selected.append(int(i)); used_q.add(q); used_r.add(r)
    return selected


def _graph_score(query, reference, qi, ri, text, selected, settings):
    if not selected:
        return 0., 0, 0
    qrows, rrows = qi[selected], ri[selected]
    mapping = np.full(len(query.tokens), -1, dtype=np.int64)
    mapping[qrows] = rrows
    similarity = np.zeros(len(query.tokens)); similarity[qrows] = text[selected]
    qneighbors = query.neighbors[qrows]
    matches = reference.neighbors[rrows][:, None, :] == mapping[qneighbors][:, :, None]
    redges = matches.argmax(2)
    ref_rows = rrows[:, None]
    relation = relation_agreement(query.relative_xy[qrows], reference.relative_xy[ref_rows, redges],
        query.relative_log_size[qrows], reference.relative_log_size[ref_rows, redges],
        query.relative_direction[qrows], reference.relative_direction[ref_rows, redges], settings)
    quality = relation * similarity[qrows][:, None] * similarity[qneighbors]
    good = matches.any(2) & (quality >= settings['minimum_relation_quality'])
    erow, ecol = np.nonzero(good)
    edges = [(int(qrows[i]), int(qneighbors[i, j])) for i, j in zip(erow, ecol)]
    # Require support from connected fragments of at least three distinct lines.
    adjacency = {int(q): set() for q in qrows}
    for q, j in edges:
        adjacency[q].add(j); adjacency[j].add(q)
    valid, seen = set(), set()
    for q in sorted(adjacency):
        if q in seen:
            continue
        component, pending = set(), [q]
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node); pending.extend(adjacency[node] - component)
        seen.update(component)
        if len(component) >= settings['minimum_matched_lines']:
            valid.update(component)
    valid_nodes = np.zeros(len(query.tokens), dtype=bool)
    valid_nodes[list(valid)] = True
    good &= valid_nodes[qrows][:, None] & valid_nodes[qneighbors]
    denominator = float(query.edge_weights.sum())
    if not good.any() or denominator <= 0:
        return 0., 0, 0
    total = float((quality * query.edge_weights[qrows])[good].sum())
    return min(1., total / denominator), len(valid), int(good.sum())


def relative_layout_score(query, reference, idf, settings):
    """Select one-to-one text matches and score their relative graph consistency."""
    minimum = settings['minimum_matched_lines']
    if len(query.tokens) < minimum or len(reference.tokens) < minimum:
        return 0., 0, 0
    pairs = matched_pairs(query, reference, idf, settings)
    if len({p[0] for p in pairs}) < minimum or len({p[1] for p in pairs}) < minimum:
        return 0., 0, 0
    qi = np.asarray([p[0] for p in pairs]); ri = np.asarray([p[1] for p in pairs])
    text = np.asarray([p[2] for p in pairs])
    lexical = np.zeros((len(query.tokens), len(reference.tokens)))
    lexical[qi, ri] = text
    qneighbors, rneighbors = query.neighbors[qi], reference.neighbors[ri]
    # Compare every neighbor relation of each proposed line correspondence.
    compatibility = relation_agreement(
        query.relative_xy[qi, :, None, :], reference.relative_xy[ri, None, :, :],
        query.relative_log_size[qi, :, None, :], reference.relative_log_size[ri, None, :, :],
        query.relative_direction[qi, :, None, :], reference.relative_direction[ri, None, :, :], settings)
    compatibility *= lexical[qneighbors[:, :, None], rneighbors[:, None, :]]
    neighbor_weights = query.weights[qneighbors]
    normalizer = neighbor_weights.sum(1)
    support = (compatibility.max(2) * neighbor_weights).sum(1) / normalizer
    best = (0., 0, 0)
    for iteration in range(settings['refinement_iterations'] + 1):
        selected = _assignment(qi, ri, text * support, query.weights, settings)
        result = _graph_score(query, reference, qi, ri, text, selected, settings)
        if result[0] > best[0]:
            best = result
        if iteration == settings['refinement_iterations'] or not selected:
            break
        mapping = np.full(len(query.tokens), -1, dtype=np.int64)
        mapping[qi[selected]] = ri[selected]
        active = rneighbors[:, None, :] == mapping[qneighbors][:, :, None]
        support = ((compatibility * active).max(2) * neighbor_weights).sum(1) / normalizer
    return best
