"""Speed and agreement against navis.models.TraversalModel on random graphs.

    python benchmarks/bench_navis.py            # CPU
    python benchmarks/bench_navis.py --gpu      # also the GPU (needs CuPy)
"""
import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

import flowlayers as fl


def random_graph(n, degree, seed=0):
    rng = np.random.default_rng(seed)
    m = n * degree
    # lognormal synapse counts, roughly like a connectome
    return pd.DataFrame({"source": rng.integers(0, n, m), "target": rng.integers(0, n, m),
                         "weight": rng.lognormal(0, 1.2, m).round() + 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sizes", default="5000:30:100,20000:50:100,50000:60:50")
    a = ap.parse_args()
    import navis
    import navis.models as nm
    navis.config.pbar_hide = True
    print("| nodes | edges | runs | navis (s) | flowlayers CPU (s) | speed-up | flowlayers GPU (s) | speed-up | mean abs diff | bias |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for spec in a.sizes.split(","):
        n, degree, runs = map(int, spec.split(":"))
        g = fl.Graph.from_edges(random_graph(n, degree))
        A = g.A.tocoo()
        edges = pd.DataFrame({"source": g.ids[A.row], "target": g.ids[A.col], "weight": A.data})
        seeds = np.arange(100)
        t = time.time()
        ref = nm.TraversalModel(edges, seeds=seeds, max_steps=15)
        with contextlib.redirect_stdout(io.StringIO()):
            ref.run(iterations=runs)
        t_navis = time.time() - t
        s_ref = ref.summary.set_index("node")["layer_mean"] if "node" in ref.summary.columns else ref.summary["layer_mean"]
        cpu = fl.TraversalModel(g, max_steps=15)
        t = time.time()
        r = cpu.run(seeds, runs=runs, seed=1)
        t_cpu = time.time() - t
        gpu_txt = "–", "–"
        if a.gpu:
            gm = fl.TraversalModel(g, max_steps=15, device="gpu")
            gm.run(seeds, runs=min(runs, 100), seed=0)            # warm-up (kernel compilation)
            t = time.time()
            gm.run(seeds, runs=runs, seed=1)
            t_gpu = time.time() - t
            gpu_txt = f"{t_gpu:.2f}", f"x{t_navis / t_gpu:.0f}"
        common = s_ref.index.intersection(r.layer_mean.dropna().index)
        d = r.layer_mean[common] - s_ref[common]
        print(f"| {n:,} | {g.n_edges:,} | {runs} | {t_navis:.1f} | {t_cpu:.2f} | x{t_navis / t_cpu:.0f} | "
              f"{gpu_txt[0]} | {gpu_txt[1]} | {d.abs().mean():.3f} | {d.mean():+.3f} |", flush=True)


if __name__ == "__main__":
    main()
