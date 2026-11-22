import numpy as np
import pandas as pd
import pytest

import flowlayers as fl
from flowlayers.exact import exact_layers


def edges_from_P(P):
    i, j = np.nonzero(P)
    # with max_w = 1 and normalize=False, the weight is the edge probability
    return pd.DataFrame({"source": i, "target": j, "weight": P[i, j]})


def mc_vs_exact(P, seeds, runs=60_000, max_steps=15, seed=0):
    model = fl.TraversalModel(edges_from_P(P), normalize=False, max_w=1.0, max_steps=max_steps)
    res = model.run(seeds, runs=runs, seed=seed, batch=5_000)
    ex = exact_layers(P, seeds, max_steps)
    return res, ex


# ------------------------------------------------------------------ exactness
def test_diamond_issue_194():
    """navis issue #194: diamond 0->1, 0->2, 1->3, 2->3, p = 0.5, seed 0. Analytic answer 3.963."""
    P = np.zeros((4, 4))
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        P[a, b] = 0.5
    res, ex = mc_vs_exact(P, [0], runs=200_000, max_steps=40)
    pasos = np.arange(ex.shape[1])
    exact_mean = (ex[3] @ pasos) / ex[3, 1:].sum()
    assert abs(exact_mean - 3.963) < 0.001
    se = np.sqrt(((pasos - exact_mean) ** 2 @ ex[3]) / ex[3, 1:].sum() / res.runs)
    assert abs(res.layer_mean[3] - exact_mean) < 4 * se


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_random_small_graphs_match_exact(seed):
    rng = np.random.default_rng(seed)
    n = 7
    P = np.where(rng.random((n, n)) < 0.35, rng.uniform(0.05, 1.0, (n, n)), 0.0)
    np.fill_diagonal(P, 0)
    P[(P > 0) & (rng.random((n, n)) < 0.2)] = 1.0          # some edges that always fire
    np.fill_diagonal(P, 0)
    res, ex = mc_vs_exact(P, [0], runs=60_000, max_steps=12)
    reach_exact = ex[:, 1:].sum(axis=1)
    np.testing.assert_allclose(res.reached.to_numpy(), reach_exact, atol=0.01)
    # full first-step distribution, node by node
    got = res.counts / res.runs
    np.testing.assert_allclose(got[:, 1:], ex[:, 1:], atol=0.01)


def test_sure_edges_are_deterministic():
    P = np.zeros((5, 5))
    for a in range(4):
        P[a, a + 1] = 1.0
    res, _ = mc_vs_exact(P, [0], runs=50)
    np.testing.assert_array_equal(res.layer_mean.to_numpy(), [1, 2, 3, 4, 5])
    assert (res.reached == 1).all()


def test_max_steps_caps_layers():
    P = np.zeros((6, 6))
    for a in range(5):
        P[a, a + 1] = 1.0
    model = fl.TraversalModel(edges_from_P(P), normalize=False, max_w=1.0, max_steps=3)
    res = model.run([0], runs=10)
    assert res.layer_mean.max() == 3
    assert np.isnan(res.layer_mean[5])


# ------------------------------------------------------------------ graph
def test_normalize_and_threshold():
    edges = pd.DataFrame({"pre": ["a", "b", "c", "a"], "post": ["d", "d", "d", "d"], "syn": [6, 2, 12, 4]})
    g = fl.Graph.from_edges(edges, source="pre", target="post", weight="syn")
    d = list(g.ids).index("d")
    assert np.isclose(g.A[:, d].sum(), 1.0)                       # duplicates summed, columns sum to 1
    assert np.isclose(g.A[list(g.ids).index("a"), d], 10 / 24)
    g5 = fl.Graph.from_edges(edges, source="pre", target="post", weight="syn", min_weight=5)
    assert g5.n_edges == 2                                         # b->d (2) and a->d (4) dropped before summing


def test_string_ids_and_missing_seeds():
    edges = pd.DataFrame({"source": ["x", "y"], "target": ["y", "z"], "weight": [1.0, 1.0]})
    model = fl.TraversalModel(edges)
    with pytest.warns(UserWarning, match="not in the graph"):
        res = model.run(["x", "nope"], runs=20)
    assert list(res.seeds) == ["x"]
    assert res.layer_mean["z"] == 3
    with pytest.raises(ValueError):
        model.run(["nope"], runs=5, missing="ignore")


def test_seed_without_out_edges_is_kept():
    """navis drops seeds that are never a source; here they stay as layer 1."""
    edges = pd.DataFrame({"source": [0], "target": [1], "weight": [1.0]})
    res = fl.TraversalModel(edges).run([0, 1], runs=5)
    assert set(res.seeds) == {0, 1}
    assert res.layer_mean[1] == 1


# ------------------------------------------------------------------ results
def test_summary_matches_target_runs():
    rng = np.random.default_rng(7)
    n, m = 300, 2_000
    edges = pd.DataFrame({"source": rng.integers(0, n, m), "target": rng.integers(0, n, m),
                          "weight": rng.integers(1, 20, m)})
    model = fl.TraversalModel(edges, max_steps=20)
    res = model.run(np.arange(5), runs=400, seed=1, targets=np.arange(n))
    lay = res.target_runs.replace(0, np.nan)
    s = res.summary
    common = s.index.intersection(lay.columns)
    np.testing.assert_allclose(s.loc[common, "layer_mean"], lay[common].mean(), rtol=1e-9)
    np.testing.assert_allclose(s.loc[common, "layer_median"], lay[common].median(), rtol=1e-9)
    np.testing.assert_allclose(s.loc[common, "layer_min"], lay[common].min())
    np.testing.assert_allclose(s.loc[common, "layer_max"], lay[common].max())
    np.testing.assert_allclose(s.loc[common, "reached"], lay[common].notna().mean())
    cmf = res.cmf()
    assert np.all(np.diff(cmf.to_numpy(), axis=1) >= -1e-12)
    np.testing.assert_allclose(cmf.iloc[:, -1].to_numpy(), res.reached.to_numpy())


def test_reproducible_and_convergence():
    rng = np.random.default_rng(3)
    n, m = 500, 5_000
    edges = pd.DataFrame({"source": rng.integers(0, n, m), "target": rng.integers(0, n, m),
                          "weight": rng.integers(1, 20, m)})
    model = fl.TraversalModel(edges)
    a = model.run(np.arange(10), runs=200, seed=42)
    b = model.run(np.arange(10), runs=200, seed=42)
    np.testing.assert_array_equal(a.counts, b.counts)
    c = model.run(np.arange(10), seed=0, tol=0.02)
    assert c.runs >= 500 and c.runs % 100 == 0 and c.change < 0.02


def test_traverse_shortcut():
    edges = pd.DataFrame({"source": [0, 1], "target": [1, 2], "weight": [1.0, 1.0]})
    res = fl.traverse(edges, [0], runs=10)
    assert res.layer_mean[2] == 3


# ------------------------------------------------------------------ optional backends
def test_gpu_matches_cpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(5)
    n, m = 3_000, 40_000
    edges = pd.DataFrame({"source": rng.integers(0, n, m), "target": rng.integers(0, n, m),
                          "weight": rng.integers(1, 20, m)})
    cpu = fl.TraversalModel(edges, device="cpu").run(np.arange(20), runs=4_000, seed=1)
    gpu = fl.TraversalModel(edges, device="gpu").run(np.arange(20), runs=4_000, seed=1)
    x, y = cpu.layer_mean.to_numpy(), gpu.layer_mean.to_numpy()
    ok = ~np.isnan(x) & ~np.isnan(y)
    assert np.corrcoef(x[ok], y[ok])[0, 1] > 0.99
    assert abs(np.mean(y[ok] - x[ok])) < 0.01


def test_same_distribution_as_navis():
    nm = pytest.importorskip("navis.models")
    import navis
    navis.config.pbar_hide = True
    rng = np.random.default_rng(11)
    n, m = 1_500, 15_000
    raw = pd.DataFrame({"source": rng.integers(0, n, m), "target": rng.integers(0, n, m),
                        "weight": rng.integers(1, 20, m)})
    g = fl.Graph.from_edges(raw)
    A = g.A.tocoo()
    edges = pd.DataFrame({"source": g.ids[A.row], "target": g.ids[A.col], "weight": A.data})
    seeds = np.arange(10)
    ref = nm.TraversalModel(edges, seeds=seeds, max_steps=15)
    ref.run(iterations=1_500)
    ours = fl.TraversalModel(g, max_steps=15).run(seeds, runs=1_500, seed=2)
    s_ref = ref.summary.set_index("node") if "node" in ref.summary.columns else ref.summary
    common = s_ref.index.intersection(ours.summary.index)
    d = ours.summary.loc[common, "layer_mean"] - s_ref.loc[common, "layer_mean"]
    assert abs(d.mean()) < 0.01                      # no bias
    assert np.abs(d).mean() < 0.05                   # differences are sampling noise
    assert set(s_ref.index) <= set(ours.summary.index)
