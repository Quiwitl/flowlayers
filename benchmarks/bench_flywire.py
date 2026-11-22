"""Whole FlyWire brain (v783): the 13 seed sets of Dorkenwald et al. (2024), with and without navis.

Needs the FlyWire connections (Zenodo 10676866) and the annotations of Schlegel et al. (2024):

    python benchmarks/bench_flywire.py --connections proofread_connections_783.feather \
        --annotations neuron_annotations.tsv --device gpu [--compare layers.parquet] [--navis-runs 2]

``--compare``: a parquet with ``root_id`` and one column per seed set (e.g. from another implementation);
prints the Spearman correlation per seed set.
"""
import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

import flowlayers as fl


def seed_sets(anot):
    c, s, sup = anot["cell_class"], anot["cell_sub_class"], anot["super_class"]
    mech = c == "mechanosensory"
    sets = {
        "olfativas": c == "olfactory", "gustativas": c == "gustatory",
        "johnston": mech & s.isin(["auditory", "wind_gravity"]), "cerdas_cabeza": mech & (s == "head bristle"),
        "cerdas_ojo": mech & (s == "eye bristle"), "taste_peg": mech & (s == "taste peg"),
        "termosensoriales": c == "thermosensory", "higrosensoriales": c == "hygrosensory",
        "proyeccion_visual": sup == "visual_projection", "fotorreceptores": (c == "visual") & (s != "ocellar"),
        "ocelos": (c == "visual") & (s == "ocellar"), "ascendentes": c == "AN",
    }
    sets["todas"] = (anot["flow"] == "afferent") | (sup == "visual_projection")
    return {k: anot.loc[v.to_numpy(), "root_id"].to_numpy() for k, v in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connections", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--min-synapses", type=float, default=None)
    ap.add_argument("--runs", type=int, default=None, help="fixed runs per set (default: until convergence)")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--navis-runs", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    t = pd.read_feather(a.connections, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    anot = pd.read_csv(a.annotations, sep="\t", usecols=["root_id", "flow", "super_class", "cell_class",
                                                          "cell_sub_class", "cell_type"])
    keep = t["pre_pt_root_id"].isin(anot["root_id"]) & t["post_pt_root_id"].isin(anot["root_id"])
    t = t[keep]
    g = fl.Graph.from_edges(sources=t["pre_pt_root_id"].to_numpy(np.int64), targets=t["post_pt_root_id"].to_numpy(np.int64),
                            weights=t["syn_count"].to_numpy(np.float64), min_weight=a.min_synapses)
    del t
    model = fl.TraversalModel(g, max_steps=a.max_steps, device=a.device)
    print(f"{g} ,  loaded and prepared in {time.time() - t0:.0f} s on {model.device}", flush=True)

    sets = seed_sets(anot)
    layers, total = {}, time.time()
    for name, seeds in sets.items():
        r = model.run(seeds, runs=a.runs, seed=783, missing="ignore")
        layers[name] = r.layer_mean
        print(f"{name:>18}: {len(r.seeds):6,} seeds, {r.runs:6,} runs, {r.seconds:6.1f} s", flush=True)
    print(f"all {len(sets)} seed sets: {(time.time() - total) / 60:.1f} min", flush=True)

    if a.compare:
        from scipy.stats import spearmanr
        ref = pd.read_parquet(a.compare)
        ref = ref.set_index(ref["root_id"].astype(np.int64))
        print("Spearman against", a.compare)
        for name, s in layers.items():
            if name in ref.columns:
                x = pd.concat([s, ref[name]], axis=1, join="inner").dropna()
                x = x[(x.iloc[:, 0] > 1) & (x.iloc[:, 1] > 1)]      # seeds excluded
                print(f"{name:>18}: rho = {spearmanr(x.iloc[:, 0], x.iloc[:, 1]).statistic:.4f}  (n = {len(x):,})")

    if a.navis_runs:
        import navis
        import navis.models as nm
        navis.config.pbar_hide = True
        A = g.A.tocoo()
        edges = pd.DataFrame({"source": g.ids[A.row], "target": g.ids[A.col], "weight": A.data})
        ref = nm.TraversalModel(edges, seeds=sets["olfativas"], max_steps=a.max_steps)
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            ref.run(iterations=a.navis_runs)
        per_run = (time.time() - t) / a.navis_runs
        ours = model.run(sets["olfativas"], runs=1000, seed=1, missing="ignore")
        print(f"olfactory, 1,000 runs: navis about {per_run * 1000 / 60:.0f} min ({per_run:.2f} s per run, "
              f"single process), flowlayers {ours.seconds:.1f} s on {model.device} "
              f"(x{per_run * 1000 / ours.seconds:.0f})")


if __name__ == "__main__":
    main()
