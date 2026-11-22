# flowlayers

Fast probabilistic traversal of connectomes: the information-flow layers of [Schlegel et al. (2021)](https://doi.org/10.7554/eLife.66018), vectorized on the CPU and optionally on the GPU.

The model is the one in [`navis.models.TraversalModel`](https://navis-org.github.io/navis/): seed neurons are layer 1, and at each step every edge from a traversed neuron to a not-yet-traversed one fires with a probability proportional to its input fraction (100 % at 0.3). `flowlayers` runs **the same Monte Carlo model** (same rules, same distribution, no approximation) but draws whole batches of runs at once as sparse matrix products, so a whole brain takes seconds per seed set instead of hours.

On the whole FlyWire brain (138,625 neurons, 15.1 million connections), the 13 sensory seed sets of Dorkenwald et al. (2024), each run until the mean layers converge (32,100 runs in total), take **8.7 minutes on a laptop GPU**. At the 3.6 s per run that `navis` takes on the same graph, that is about 32 hours.

## Install

```bash
pip install flowlayers            # CPU (numpy, scipy, pandas)
pip install "flowlayers[gpu]"     # + CuPy for NVIDIA GPUs (CUDA 12)
```

## Use

```python
import flowlayers as fl

# edges: a DataFrame with presynaptic ID, postsynaptic ID and synapse count
model = fl.TraversalModel(edges, source="pre", target="post", weight="syn_count",
                          max_steps=15, device="auto")      # "cpu", "gpu" or "auto"
res = model.run(olfactory_ids)          # batches of 100 runs until the mean layer converges

res.layer_mean        # pd.Series: mean layer of every neuron (NaN if never reached)
res.summary           # layer_min, layer_max, layer_mean, layer_median, reached  (navis' summary columns)
res.cmf()             # probability of having been reached by each step
res.reached           # fraction of runs in which each neuron was reached
```

- **Weights.** By default the weights are synapse counts and are turned into input fractions (`normalize=True`). Pass `normalize=False` if they already are input fractions, as `navis` expects. `min_weight=5` drops edges with fewer than 5 synapses *before* normalizing — this changes the layers noticeably (see below), so report it.
- **Runs.** `run(seeds, runs=1000)` does exactly 1,000 runs. Without `runs`, batches of 100 are added until the mean layer changes by less than `tol=0.01` in 99 % of the neurons (at least 5 batches; `max_runs=10_000`).
- **Per-run layers.** `run(seeds, targets=motor_neuron_ids)` also keeps the layer of those neurons in every run (`res.target_runs`), e.g. for bootstrap intervals.
- **Reuse.** Build the model once and call `run` for each seed set; the graph is prepared only once (and kept on the GPU).
- **Probability.** `max_w=0.3` and `min_w=0` as in `navis` (`linear_activation_p`).

## How it works

Drawing each edge independently and traversing a neuron if any of its edges fires is the same as drawing each neuron once with

P<sub>j</sub> = 1 − Π<sub>i</sub> (1 − p<sub>ij</sub>)  over the traversed *i*,

which is `1 − exp(Σ log(1 − p_ij))`: one sparse matrix product with the indicator matrix of traversed neurons. Each run is a column of that matrix, so one step of a whole batch of runs is a single sparse × dense product (scipy on the CPU, cuSPARSE through CuPy on the GPU; the same code runs on both). Failed neurons are tried again at later steps, as in `navis`. Results are kept as the full distribution of the first-traversal step of every neuron, from which the mean, median, min, max and CMF are derived.

## Validation

- **Exact answer on small graphs.** `flowlayers.exact.exact_layers` enumerates every set of traversed nodes and gives the exact first-traversal distribution. On the diamond of [navis issue #194](https://github.com/navis-org/navis/issues/194) (0→1, 0→2, 1→3, 2→3, p = 0.5) it gives 3.96296 for node 3, the analytic value reported there; `flowlayers` matches it, and matches the full distribution node by node on random small graphs.
- **Same distribution as navis.** On random graphs the mean layers agree with `navis.models.TraversalModel` with no bias; the remaining differences are sampling noise.
- **Published layers.** This model, on the whole brain with no synapse threshold and a 15-step cap, reproduces the layers published by Dorkenwald et al. (2024) for FlyWire v783 in all 13 seed sets (Spearman 0.971–0.996, [fly-sensory-layers](https://github.com/Quiwitl/fly-sensory-layers)); flowlayers matches that implementation at ρ ≥ 0.998 (below).
- **CPU and GPU.** Same results on both devices (tested).

## Benchmarks

Laptop with an AMD Ryzen 7 260 (16 threads) and an NVIDIA RTX 5050 Laptop GPU (8 GB), Windows 11, navis 1.12.0, numpy 2.5, scipy 1.18, CuPy 14.2. Scripts in `benchmarks/`.

**Random graphs** (log-normal synapse counts, 20 seeds, 100 runs, `max_steps=15`; `bench_navis.py --gpu`). "Mean abs diff" and "bias" compare the mean layers of flowlayers and navis; with 100 runs each the absolute difference is sampling noise, and the bias stays below 0.01 layers.

| nodes | edges | navis | flowlayers CPU | speed-up | flowlayers GPU | speed-up | mean abs diff | bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 149,557 | 2.0 s | 0.14 s | ×14 | 0.02 s | ×86 | 0.092 | +0.001 |
| 20,000 | 998,750 | 12.2 s | 1.32 s | ×9 | 0.11 s | ×116 | 0.102 | −0.008 |
| 50,000 | 2,998,196 | 42.4 s | 8.57 s | ×5 | 0.25 s | ×168 | 0.107 | −0.006 |
| 140,000 | 13,994,879 | 143.6 s | 39.7 s | ×4 | 1.28 s | ×112 | 0.110 | −0.005 |

The CPU gain shrinks on big graphs (the sparse product becomes memory-bound); the GPU is the option for whole brains.

**Whole FlyWire brain** (v783, proofread connections, no synapse threshold, 15 steps; `bench_flywire.py --device gpu --navis-runs 2`):

| | navis | flowlayers GPU |
|---|---:|---:|
| 1,000 runs from the olfactory receptor neurons | ≈ 60 min (3.61 s per run) | 16.5 s (×219) |
| 13 seed sets to convergence (32,100 runs) | ≈ 32 h (extrapolated) | 8.7 min |

The mean layers agree with the independent implementation in [fly-sensory-layers](https://github.com/Quiwitl/fly-sensory-layers) at Spearman ρ = 0.998–0.9998 for every seed set (seeds excluded), the level of run-to-run noise.

## Differences from navis

- Seeds that never appear as a source are **kept** (as layer 1). `navis` drops them silently, which turns them into ordinary nodes.
- A run stops when no edge with p > 0 is left (or at `max_steps`); `navis` also waits on edges with p = 0, which changes when it stops but not the layers.
- Only the linear probability function (`min_w`, `max_w`) is supported; arbitrary per-edge Python callables cannot be vectorized.
- No `results` table with one row per node and run (it would have hundreds of millions of rows for a whole brain); the per-step counts hold the same information.

## Cite

If you use `flowlayers`, please cite the model:

> Schlegel P., Bates A. S., Stürner T., *et al.* (2021). Information flow, cell types and stereotypy in a full olfactory connectome. *eLife* 10, e66018.

and this repository. MIT license.
