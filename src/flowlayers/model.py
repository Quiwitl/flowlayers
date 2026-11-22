"""Probabilistic traversal model (Schlegel et al. 2021), vectorized on CPU and GPU."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .graph import Graph

__all__ = ["TraversalModel", "TraversalResult", "traverse", "linear_p"]


def linear_p(w, min_w: float = 0.0, max_w: float = 0.3):
    """Edge probability: 0 at ``min_w`` rising linearly to 1 at ``max_w`` (the ``navis`` default)."""
    return np.clip((np.asarray(w, dtype=np.float64) - min_w) / (max_w - min_w), 0.0, 1.0)


def _backend(device: str):
    """Array and sparse modules for ``device`` ("cpu", "gpu" or "auto")."""
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError('device must be "cpu", "gpu" or "auto"')
    if device in ("gpu", "auto"):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            cp.cuda.runtime.getDeviceCount()
            return cp, cps, "gpu"
        except Exception as exc:  # no CuPy, no driver or no device
            if device == "gpu":
                raise RuntimeError(f"device='gpu' needs CuPy and a CUDA device: {exc}") from exc
    return np, sparse, "cpu"


@dataclass
class TraversalResult:
    """First-traversal step of every node over all runs.

    ``counts[k, t]`` is the number of runs in which node ``k`` was first reached at step ``t``
    (``t = 1`` are the seeds; column 0 is unused). Everything else is derived from it.
    """

    ids: np.ndarray
    counts: np.ndarray
    runs: int
    seeds: np.ndarray
    max_steps: int
    change: float = float("nan")
    seconds: float = float("nan")
    device: str = "cpu"
    target_runs: pd.DataFrame | None = field(default=None, repr=False)

    # -------------------------------------------------------------- per node
    @property
    def n_reached(self) -> np.ndarray:
        return self.counts[:, 1:].sum(axis=1)

    @property
    def reached(self) -> pd.Series:
        """Fraction of runs in which each node was reached."""
        return pd.Series(self.n_reached / self.runs, index=self.ids, name="reached")

    @property
    def layer_mean(self) -> pd.Series:
        """Mean first-traversal step over the runs that reached the node (NaN if never reached)."""
        pasos = np.arange(self.counts.shape[1])
        n = self.n_reached
        media = np.divide(self.counts @ pasos, n, out=np.full(len(n), np.nan), where=n > 0)
        return pd.Series(media, index=self.ids, name="layer_mean")

    def cmf(self, conditional: bool = False) -> pd.DataFrame:
        """Cumulative probability of having been reached by each step (columns 1..max_steps).

        With ``conditional=True`` it is conditioned on the node being reached at all."""
        c = np.cumsum(self.counts[:, 1:], axis=1).astype(np.float64)
        den = self.n_reached[:, None] if conditional else self.runs
        c = np.divide(c, den, out=np.zeros_like(c), where=np.broadcast_to(den, c.shape) > 0)
        return pd.DataFrame(c, index=self.ids, columns=np.arange(1, self.counts.shape[1]))

    @property
    def summary(self) -> pd.DataFrame:
        """``layer_min``, ``layer_max``, ``layer_mean``, ``layer_median`` and ``reached`` for every node
        reached at least once: the columns of ``navis.models.TraversalModel.summary``."""
        n = self.n_reached
        ok = n > 0
        c = self.counts[ok, 1:]
        pasos = np.arange(1, self.counts.shape[1])
        hay = c > 0
        layer_min = pasos[hay.argmax(axis=1)]
        layer_max = pasos[c.shape[1] - 1 - hay[:, ::-1].argmax(axis=1)]
        acum = np.cumsum(c, axis=1)
        half = n[ok] / 2.0
        lo = pasos[(acum >= half[:, None]).argmax(axis=1)]             # lower median
        hi = pasos[(acum > half[:, None]).argmax(axis=1)]              # upper median
        median = np.where(n[ok] % 2 == 0, (lo + hi) / 2.0, hi).astype(np.float64)
        return pd.DataFrame({"layer_min": layer_min, "layer_max": layer_max,
                             "layer_mean": self.layer_mean.to_numpy()[ok], "layer_median": median,
                             "reached": n[ok] / self.runs}, index=pd.Index(self.ids[ok], name="node"))

    def __repr__(self) -> str:
        return (f"<flowlayers.TraversalResult: {len(self.ids):,} nodes, {int((self.n_reached > 0).sum()):,} reached, "
                f"{self.runs:,} runs on {self.device}, {self.seconds:.1f} s>")


class TraversalModel:
    """The traversal model of Schlegel et al. (2021), as in ``navis.models.TraversalModel``.

    Seeds are layer 1. At each step, every edge ``i -> j`` from a traversed node ``i`` to a node ``j`` that
    is not yet traversed succeeds with probability ``p_ij = clip((A_ij - min_w) / (max_w - min_w), 0, 1)``;
    ``j`` is traversed if any of its edges succeeds and gets that step as its layer. Nodes that fail are
    tried again at later steps. A run ends when no candidate edges are left or at ``max_steps``.

    Drawing once per node with ``P_j = 1 - prod_i (1 - p_ij)`` has the same distribution as drawing per
    edge, and turns each step of a batch of runs into one sparse-times-dense product.

    Parameters
    ----------
    graph : Graph or DataFrame
        A :class:`Graph`, or an edge list passed to :meth:`Graph.from_edges` with ``graph_kwargs``.
    max_w, min_w : float
        Input fraction at which an edge always / never fires (navis: 0.3 and 0).
    max_steps : int
        Step cap per run (navis default 15).
    device : "cpu", "gpu" or "auto"
        "gpu" needs CuPy; "auto" uses the GPU if it is available.
    """

    def __init__(self, graph, *, max_w: float = 0.3, min_w: float = 0.0, max_steps: int = 15,
                 device: str = "cpu", **graph_kwargs):
        if not isinstance(graph, Graph):
            graph = Graph.from_edges(graph, **graph_kwargs)
        elif graph_kwargs:
            raise TypeError(f"unexpected arguments for an existing Graph: {sorted(graph_kwargs)}")
        if not 0 <= min_w < max_w:
            raise ValueError("need 0 <= min_w < max_w")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.graph, self.max_w, self.min_w, self.max_steps = graph, max_w, min_w, int(max_steps)
        self.xp, self.sp, self.device = _backend(device)

        A = graph.A
        p = A.copy()
        p.data = linear_p(p.data, min_w, max_w)
        p.eliminate_zeros()
        sure = p.copy()
        sure.data = (sure.data >= 1.0).astype(np.float32)
        sure.eliminate_zeros()
        logq = p.copy()
        logq.data = np.where(p.data < 1.0, np.log1p(-np.minimum(p.data, 1 - 1e-12)), 0.0)
        logq.eliminate_zeros()
        cand = p.copy()
        cand.data = np.ones_like(cand.data, dtype=np.float32)
        # transposed: row = target node; on the GPU the matrices live on the card
        to = (lambda M: self.sp.csr_matrix(sparse.csr_matrix(M.T, dtype=np.float32))) if self.device == "gpu" \
            else (lambda M: sparse.csr_matrix(M.T, dtype=np.float32))
        self._logqT, self._sureT, self._candT = to(logq), to(sure), to(cand)

    def __repr__(self) -> str:
        return (f"<flowlayers.TraversalModel: {self.graph.n_nodes:,} nodes, {self.graph.n_edges:,} edges, "
                f"max_w={self.max_w}, max_steps={self.max_steps}, device={self.device}>")

    # -------------------------------------------------------------- one batch
    def _batch(self, seeds_idx, runs: int, rng):
        """Layers of ``runs`` independent runs, as int16 (n, runs) on the device; 0 = not reached."""
        xp = self.xp
        n = self.graph.n_nodes
        layers = xp.zeros((n, runs), dtype=xp.int16)
        X = xp.zeros((n, runs), dtype=xp.float32)
        layers[seeds_idx, :] = 1
        X[seeds_idx, :] = 1.0
        for t in range(2, self.max_steps + 1):
            pending = (layers == 0) & ((self._candT @ X) > 0)
            if not bool(pending.any()):
                break
            P = -xp.expm1(self._logqT @ X)            # 1 - prod(1 - p) over edges with p < 1
            P[(self._sureT @ X) > 0] = 1.0            # any edge with p = 1
            new = pending & (rng.random((n, runs), dtype=xp.float32) < P)
            del P
            layers[new] = t
            X[new] = 1.0
        return layers

    # -------------------------------------------------------------- run
    def run(self, seeds, *, runs: int | None = None, batch: int = 100, tol: float = 0.01,
            quantile: float = 0.99, max_runs: int = 10_000, seed=None, targets=None,
            missing: str = "warn", progress=None) -> TraversalResult:
        """Traverse from ``seeds`` and return a :class:`TraversalResult`.

        Parameters
        ----------
        seeds : iterable of node IDs
            Seeds missing from the graph are dropped with a warning (``missing``). Unlike navis, seeds
            without outgoing edges are kept (they are layer 1 and reach nothing).
        runs : int, optional
            Exact number of runs. If None (default), batches of ``batch`` runs are added until the mean
            layer changes by less than ``tol`` in the ``quantile`` of nodes (at least 5 batches), up to
            ``max_runs``.
        seed : int or numpy/cupy Generator, optional
            Random seed for reproducibility.
        targets : iterable of node IDs, optional
            Keep the layer of these nodes in every run (``result.target_runs``), e.g. for bootstraps.
        progress : callable, optional
            Called as ``progress(runs_done, change)`` after every batch.
        """
        xp = self.xp
        t0 = time.time()
        idx = self.graph.index(seeds, missing=missing)
        if len(idx) == 0:
            raise ValueError("none of the seeds are in the graph")
        seeds_idx = xp.asarray(idx)
        tgt_idx = self.graph.index(targets, missing=missing) if targets is not None else None
        rng = seed if hasattr(seed, "random") else xp.random.default_rng(seed)
        n, S = self.graph.n_nodes, self.max_steps
        counts = xp.zeros((n, S + 1), dtype=xp.int64)
        kept = []
        fixed = runs is not None
        total = int(runs) if fixed else int(max_runs)
        done, change, prev = 0, float("inf"), None
        pasos = xp.arange(S + 1)
        while done < total:
            k = min(batch, total - done)
            L = self._batch(seeds_idx, k, rng)
            for t in range(1, S + 1):
                counts[:, t] += (L == t).sum(axis=1)
            if tgt_idx is not None:
                kept.append(L[xp.asarray(tgt_idx)].get() if xp is not np else L[tgt_idx].copy())
            del L
            done += k
            if not fixed:
                c = counts[:, 1:].sum(axis=1)
                mean = xp.where(c > 0, (counts @ pasos) / xp.maximum(c, 1), xp.nan)
                mean = mean.get() if xp is not np else mean
                if prev is not None:
                    ok = ~np.isnan(mean) & ~np.isnan(prev)
                    change = float(np.quantile(np.abs(mean[ok] - prev[ok]), quantile)) if ok.any() else 0.0
                prev = mean
            if progress:
                progress(done, change)
            if not fixed and done >= 5 * batch and change < tol:
                break
        counts = counts.get() if xp is not np else counts
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        if not fixed and change >= tol:
            warnings.warn(f"did not converge in {done:,} runs (change {change:.4f} >= tol {tol})", stacklevel=2)
        target_runs = None
        if tgt_idx is not None:
            target_runs = pd.DataFrame(np.concatenate(kept, axis=1).T, columns=self.graph.ids[tgt_idx])
            target_runs.index.name = "run"
        return TraversalResult(ids=self.graph.ids, counts=counts, runs=done, seeds=self.graph.ids[idx],
                               max_steps=S, change=change, seconds=time.time() - t0, device=self.device,
                               target_runs=target_runs)


def traverse(edges, seeds, *, max_w: float = 0.3, min_w: float = 0.0, max_steps: int = 15,
             device: str = "cpu", graph_kwargs: dict | None = None, **run_kwargs) -> TraversalResult:
    """One-call shortcut: ``TraversalModel(edges, ...).run(seeds, ...)``."""
    model = TraversalModel(edges, max_w=max_w, min_w=min_w, max_steps=max_steps, device=device,
                           **(graph_kwargs or {}))
    return model.run(seeds, **run_kwargs)
