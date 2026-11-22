"""Exact first-traversal distribution for very small graphs, by enumerating every set of traversed nodes.

Used to test the Monte Carlo engine. The cost grows as 2**n, so keep n <= ~12.
"""
from __future__ import annotations

from itertools import product

import numpy as np

__all__ = ["exact_layers"]


def exact_layers(P: np.ndarray, seeds, max_steps: int = 15) -> np.ndarray:
    """``out[k, t]`` = probability that node ``k`` is first reached at step ``t`` (``t = 1..max_steps``).

    ``P[i, j]`` is the edge probability ``i -> j`` (0 if there is no edge). Same rules as the model:
    at each step every untraversed node ``j`` is reached with probability ``1 - prod_i (1 - P[i, j])``
    over the traversed ``i``, independently of the other nodes; failed nodes are retried.
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]
    start = frozenset(int(s) for s in seeds)
    out = np.zeros((n, max_steps + 1))
    for s in start:
        out[s, 1] = 1.0
    states = {start: 1.0}
    for t in range(2, max_steps + 1):
        nxt: dict[frozenset, float] = {}
        for S, pr in states.items():
            cand = [j for j in range(n) if j not in S]
            pj = [1.0 - np.prod([1.0 - P[i, j] for i in S]) for j in cand]
            live = [(j, q) for j, q in zip(cand, pj) if q > 0]
            if not live:
                nxt[S] = nxt.get(S, 0.0) + pr
                continue
            for hits in product((0, 1), repeat=len(live)):
                w = pr
                for (j, q), h in zip(live, hits):
                    w *= q if h else 1.0 - q
                if w == 0.0:
                    continue
                new = [j for (j, _), h in zip(live, hits) if h]
                for j in new:
                    out[j, t] += w
                T = S | frozenset(new)
                nxt[T] = nxt.get(T, 0.0) + w
        states = nxt
    return out
