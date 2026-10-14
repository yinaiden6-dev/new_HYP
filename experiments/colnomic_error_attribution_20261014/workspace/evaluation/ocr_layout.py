"""Partial OCR-line geometry matching, independent of package labels and images."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import numpy as np

from .scoring import tokenize


@dataclass
class Lines:
    centers: np.ndarray
    directions: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    tokens: list
    weights: np.ndarray
    postings: dict


def reference_idf(records):
    """Use the same document-level IDF definition and tokenizer as BM25."""
    counts = Counter(token for record in records
                     for token in set(tokenize(record['text'])))
    n = len(records)
    return {token: math.log(1 + (n - count + .5) / (count + .5))
            for token, count in counts.items()}


def extract_lines(record, idf):
    centers, directions, widths, heights, tokens, weights = [], [], [], [], [], []
    for line in record.get('lines', []):
        words = set(tokenize(line.get('text', ''))) & idf.keys()
        polygon = np.asarray(line.get('polygon', []), dtype=np.float64)
        if not words or polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            continue
        direction = polygon[1] - polygon[0]
        width = float(np.linalg.norm(direction))
        if width < 1e-6:
            continue
        # Perpendicular separation also works for skewed quadrilaterals.
        unit = direction / width
        side = ((polygon[3] - polygon[0]) + (polygon[2] - polygon[1])) / 2
        height = abs(float(unit[0] * side[1] - unit[1] * side[0]))
        if height < 1e-6:
            continue
        centers.append(polygon.mean(0)); directions.append(unit)
        widths.append(width); heights.append(height); tokens.append(words)
        weights.append(sum(idf[word] for word in sorted(words)))
    postings = defaultdict(list)
    for i, words in enumerate(tokens):
        for word in sorted(words):
            postings[word].append(i)
    return Lines(np.asarray(centers).reshape(-1, 2),
                 np.asarray(directions).reshape(-1, 2),
                 np.asarray(widths), np.asarray(heights), tokens,
                 np.asarray(weights), dict(postings))


def matched_pairs(query, reference, idf, settings):
    pairs = []
    for qi, words in enumerate(query.tokens):
        possible = {ri for word in words for ri in reference.postings.get(word, [])}
        for ri in sorted(possible):
            overlap = sum(idf[word] for word in sorted(words & reference.tokens[ri]))
            similarity = 2 * overlap / (query.weights[qi] + reference.weights[ri])
            if similarity >= settings['minimum_text_similarity']:
                pairs.append((qi, ri, similarity))
    pairs.sort(key=lambda p: (-p[2] * query.weights[p[0]], p[0], p[1]))
    return pairs[:settings['max_line_pairs']]


def layout_score(query, reference, idf, settings):
    """Best local, one-to-one line support under an oriented similarity transform.

    A text-line pair proposes rotation, scale (from line height), and translation.
    Other lines must agree in position, size, and reading direction. The best
    fragment may occur anywhere in an unfolded reference. No reflection is
    allowed. Missing/insufficient geometry supplies no bonus.
    """
    pairs = matched_pairs(query, reference, idf, settings)
    minimum = settings['minimum_matched_lines']
    if (len(query.tokens) < minimum or len(pairs) < minimum or
            len({p[0] for p in pairs}) < minimum or
            len({p[1] for p in pairs}) < minimum):
        return 0., 0
    qi = np.array([p[0] for p in pairs])
    ri = np.array([p[1] for p in pairs])
    text = np.array([p[2] for p in pairs])
    anchors = np.arange(min(len(pairs), settings['max_hypotheses']))
    aq, ar = qi[anchors], ri[anchors]
    qd, rd = query.directions[aq], reference.directions[ar]
    cosine = (qd * rd).sum(1)
    sine = qd[:, 0] * rd[:, 1] - qd[:, 1] * rd[:, 0]
    scale = reference.heights[ar] / query.heights[aq]
    # Rotate each matched line's center relative to each proposed anchor.
    delta = query.centers[qi][None, :, :] - query.centers[aq][:, None, :]
    predicted = np.stack((cosine[:, None] * delta[:, :, 0] - sine[:, None] * delta[:, :, 1],
                          sine[:, None] * delta[:, :, 0] + cosine[:, None] * delta[:, :, 1]), axis=2)
    predicted = predicted * scale[:, None, None] + reference.centers[ar][:, None, :]
    transformed_height = scale[:, None] * query.heights[qi][None, :]
    height = reference.heights[ri][None, :]
    distance = np.linalg.norm(predicted - reference.centers[ri][None, :, :], axis=2)
    distance /= settings['distance_sigma_heights'] * np.maximum(height, transformed_height)
    size_sigma = settings['log_size_sigma']
    height_error = np.log(transformed_height / height) / size_sigma
    width_error = np.log(scale[:, None] * query.widths[qi][None, :] /
                         reference.widths[ri][None, :]) / size_sigma
    direction = query.directions[qi]
    rotated = np.stack((cosine[:, None] * direction[None, :, 0] - sine[:, None] * direction[None, :, 1],
                        sine[:, None] * direction[None, :, 0] + cosine[:, None] * direction[None, :, 1]), axis=2)
    angle = np.maximum(0., (rotated * reference.directions[ri][None, :, :]).sum(2)) ** 2
    quality = (text[None, :] * angle *
               np.exp(-.5 * (distance ** 2 + height_error ** 2 + width_error ** 2)))
    weighted = quality * query.weights[qi][None, :]
    # First estimate support without exclusivity; refine only the best hypotheses.
    support = np.zeros((len(anchors), len(query.tokens)))
    for i in range(len(query.tokens)):
        indices = np.flatnonzero(qi == i)
        if len(indices):
            support[:, i] = weighted[:, indices].max(1)
    best_anchors = np.argsort(-support.sum(1), kind='stable')[:settings['refine_hypotheses']]
    best, best_count = 0., 0
    for a in best_anchors:
        used_q, used_r, total, count = set(), set(), 0., 0
        for i in np.argsort(-weighted[a], kind='stable'):
            q, r = int(qi[i]), int(ri[i])
            if q not in used_q and r not in used_r and quality[a, i] >= .1:
                used_q.add(q); used_r.add(r)
                total += float(weighted[a, i]); count += 1
        if count >= minimum:
            score = total / float(query.weights.sum())
            if score > best:
                best, best_count = score, count
    return min(1., best), best_count
