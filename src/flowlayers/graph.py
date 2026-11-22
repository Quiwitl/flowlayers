"""Turn an edge list into the sparse matrices the traversal needs."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = ["Graph"]


class Graph:
    """A directed, weighted graph with an input-fraction matrix ``A``.

    ``A[i, j]`` is the fraction of the inputs of ``j`` that come from ``i``, which is what the
    traversal model of Schlegel et al. (2021) turns into an edge probability.

    Use :meth:`from_edges` to build it from a DataFrame (or arrays) of ``source, target, weight``.
    Node IDs can be any hashable type; internally they are mapped to ``0..n-1``.
    """

    def __init__(self, A: sparse.csr_matrix, ids: np.ndarray):
        A = sparse.csr_matrix(A, dtype=np.float64)
        if A.shape[0] != A.shape[1] or A.shape[0] != len(ids):
            raise ValueError("A must be square with one row per id")
        self.A = A
        self.ids = np.asarray(ids)
        self._index = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_edges(cls, edges: pd.DataFrame | None = None, *, source: str = "source", target: str = "target",
                   weight: str | None = "weight", normalize: bool = True, min_weight: float | None = None,
                   sources=None, targets=None, weights=None) -> "Graph":
        """Build a graph from an edge list.

        Parameters
        ----------
        edges : DataFrame, optional
            Edge list with ``source``, ``target`` and (optionally) ``weight`` columns. Alternatively pass
            ``sources``, ``targets`` and ``weights`` as arrays.
        weight : str or None
            Weight column. ``None`` gives every edge weight 1.
        normalize : bool
            If True (default), weights are synapse counts (or any non-negative strength) and are divided by
            the total input of each target, giving input fractions. If False, weights are used as they are
            (they should already be input fractions, as in ``navis.models.TraversalModel``).
        min_weight : float, optional
            Drop edges with weight below this value *before* normalizing (e.g. ``min_weight=5`` for the
            usual 5-synapse threshold). The threshold changes the layers noticeably, so report it.

        Duplicate edges are summed; self-loops are dropped (they never change a traversal).
        """
        if edges is not None:
            if not isinstance(edges, pd.DataFrame):
                raise TypeError(f"edges must be a pandas DataFrame, got {type(edges)}")
            for c in (source, target) + ((weight,) if weight else ()):
                if c not in edges.columns:
                    raise KeyError(f'edges has no "{c}" column')
            sources = edges[source].to_numpy()
            targets = edges[target].to_numpy()
            weights = edges[weight].to_numpy(dtype=np.float64) if weight else None
        if sources is None or targets is None:
            raise ValueError("pass an edge DataFrame or sources and targets arrays")
        sources, targets = np.asarray(sources), np.asarray(targets)
        weights = np.ones(len(sources)) if weights is None else np.asarray(weights, dtype=np.float64)
        if not (len(sources) == len(targets) == len(weights)):
            raise ValueError("sources, targets and weights must have the same length")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        keep = sources != targets
        if min_weight is not None:
            keep &= weights >= min_weight
        sources, targets, weights = sources[keep], targets[keep], weights[keep]

        ids, inv = np.unique(np.concatenate([sources, targets]), return_inverse=True)
        n, m = len(ids), len(sources)
        i, j = inv[:m], inv[m:]
        W = sparse.csr_matrix((weights, (i, j)), shape=(n, n))  # duplicates are summed
        W.eliminate_zeros()
        if normalize:
            W = W.tocsc()
            entradas = np.asarray(W.sum(axis=0)).ravel()
            inv_in = np.divide(1.0, entradas, out=np.zeros_like(entradas), where=entradas > 0)
            W = sparse.csr_matrix(W @ sparse.diags(inv_in))
        elif W.nnz and W.data.max() > 1:
            warnings.warn("normalize=False but some weights are > 1: they should be input fractions", stacklevel=2)
        return cls(W, ids)

    # ------------------------------------------------------------------ ids
    @property
    def n_nodes(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return int(self.A.nnz)

    def index(self, nodes, missing: str = "warn") -> np.ndarray:
        """Indices of ``nodes`` in the graph. ``missing``: "warn" (drop with a warning), "raise" or "ignore"."""
        nodes = np.unique(np.asarray(list(nodes) if not isinstance(nodes, np.ndarray) else nodes))
        pos = np.searchsorted(self.ids, nodes)
        pos = np.clip(pos, 0, max(len(self.ids) - 1, 0))
        found = (len(self.ids) > 0) & (self.ids[pos] == nodes) if len(nodes) else np.zeros(0, bool)
        if not found.all():
            n_miss = int((~found).sum())
            if missing == "raise":
                raise KeyError(f"{n_miss} node(s) are not in the graph")
            if missing == "warn":
                warnings.warn(f"{n_miss} of {len(nodes)} node(s) are not in the graph and were dropped", stacklevel=3)
        return pos[found]

    def __repr__(self) -> str:
        return f"<flowlayers.Graph: {self.n_nodes:,} nodes, {self.n_edges:,} edges>"
