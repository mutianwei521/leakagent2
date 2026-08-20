"""
Apples-to-apples baseline comparison: trained localizers are trained on the KB
and evaluated on the SAME test leaks the executor-supervisor system is evaluated
on (random + representative nodes, sigma=0.05, the five seeds). This is the fair
forced (coverage=1) comparison; it shows that a trained localizer generalizes
poorly to off-library leak nodes under field noise, just as retrieval does, while
the system's twin-fit does not.

Run:  python eval/baselines_fair.py     (via the wds_rag interpreter)
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from tools.graphrag_tool import GraphRAGTool
from eval.run_experiments import generate
from eval.baselines import load_xy

NOISE, SEEDS = 0.05, [17, 42, 101, 2025, 31337]


def build_gcn_predictor(Xkb, ykb, sensors, epochs=150):
    """Fair-protocol GCN: a genuinely-trained 3-layer graph convolutional network
    trained ONCE on the KB fingerprints (per-scenario sensor graph, graph-level zone
    classification), returning predict(Xte) -> zone-id array for the system test leaks."""
    import torch
    import torch.nn.functional as Fnn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    torch.manual_seed(42)
    np.random.seed(42)
    # Pin CPU for a fully deterministic, reproducible GCN number: CUDA scatter/gather
    # in GCNConv is nondeterministic, so a GPU run drifts by ~1% between repeats. The
    # model is tiny (per-scenario sensor graph), so CPU training is cheap and exact.
    dev = torch.device("cpu")
    topo = json.load(open(os.path.join(ROOT, "data", "exa7", "topology_index.json"), encoding="utf-8"))
    n2p = {str(k): int(v) for k, v in topo.get("node_to_partition", {}).items()}
    adjd = json.load(open(os.path.join(ROOT, "data", "exa7", "partition_adjacency.json"), encoding="utf-8"))
    adj = adjd.get("adjacency", adjd)
    nbr = {int(k): set(int(x) for x in v) for k, v in adj.items()} if isinstance(adj, dict) else {}
    sp = [n2p.get(s, -1) for s in sensors]
    ei = [[i, j] for i in range(len(sensors)) for j in range(len(sensors))
          if i != j and (sp[i] == sp[j] or sp[j] in nbr.get(sp[i], set()))]
    if not ei:
        ei = [[i, j] for i in range(len(sensors)) for j in range(len(sensors)) if i != j]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    classes = sorted(set(int(v) for v in ykb))
    cidx = {c: i for i, c in enumerate(classes)}

    def graphs_of(X, Y=None):
        return [Data(x=torch.tensor(X[k], dtype=torch.float).view(-1, 1), edge_index=edge_index,
                     y=(torch.tensor([cidx.get(int(Y[k]), 0)], dtype=torch.long) if Y is not None else None))
                for k in range(len(X))]

    class GCN(torch.nn.Module):
        def __init__(self, h=64, c=len(classes)):
            super().__init__()
            self.l = torch.nn.Linear(1, h)
            self.c1, self.c2, self.c3 = GCNConv(h, h), GCNConv(h, h), GCNConv(h, h)
            self.head = torch.nn.Linear(h, c)

        def forward(self, d):
            x = Fnn.relu(self.l(d.x))
            x = Fnn.relu(self.c1(x, d.edge_index))
            x = Fnn.relu(self.c2(x, d.edge_index))
            x = Fnn.relu(self.c3(x, d.edge_index))
            return self.head(global_mean_pool(x, d.batch))

    model = GCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trl = DataLoader(graphs_of(Xkb, ykb), batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in trl:
            b = b.to(dev); opt.zero_grad()
            loss = Fnn.cross_entropy(model(b), b.y); loss.backward(); opt.step()
    model.eval()

    def predict(Xte):
        tel = DataLoader(graphs_of(Xte), batch_size=64)
        out = []
        with torch.no_grad():
            for b in tel:
                out.append(model(b.to(dev)).argmax(1).cpu().numpy())
        idx = np.concatenate(out) if out else np.array([], dtype=int)
        return np.array([classes[i] for i in idx])

    return predict


def main():
    print("=" * 70)
    print("FAIR FORCED-LOCALIZER COMPARISON (train on KB, test on system test leaks)")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    sensors = ctx.sensors

    # train classical baselines on the KB fingerprints (noise-augmented)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xkb, ykb, _, _ = load_xy(noise=NOISE, seed=0)
    models = {
        "kNN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine"),
        "RandomForest (200)": RandomForestClassifier(n_estimators=200, random_state=42),
        "MLP (128,64)": make_pipeline(StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                                    early_stopping=True, random_state=42)),
    }
    for clf in models.values():
        clf.fit(Xkb, ykb)
    print("training GCN on KB (fair protocol) ...", flush=True)
    gcn_predict = build_gcn_predictor(Xkb, ykb, sensors)

    # build the same test set (leaks only) across seeds, evaluate everyone forced
    acc = {k: [] for k in list(models) + ["GraphRAG retrieval", "GCN (3-layer, trained)",
                                          "Executor-Supervisor (twin-fit)"]}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        leaks = [s for s in generate(ctx, gr, rng, NOISE) if s["true_class"] == "leak"]
        Xte = np.array([[s["observation"].get(se, 0.0) for se in sensors] for s in leaks])
        yte = np.array([s["true_region"] for s in leaks])
        for name, clf in models.items():
            acc[name].append(float((clf.predict(Xte) == yte).mean()))
        acc["GCN (3-layer, trained)"].append(float((gcn_predict(Xte) == yte).mean()))
        rt = [gr.rank_partitions(s["observation"], top_k=1)[0][0] for s in leaks]
        acc["GraphRAG retrieval"].append(float((np.array(rt) == yte).mean()))

    # system twin-fit number from the paper-grade run
    rf = json.load(open(os.path.join(ROOT, "artifacts", "results_full.json"), encoding="utf-8"))
    sysacc = rf["multiseed_sigma0.05"]["leak_partition_acc_with_active"]
    acc["Executor-Supervisor (twin-fit)"] = sysacc["values"]

    print(f"\nforced Top-1 on identical test leaks (mean ± s.d. over {len(SEEDS)} seeds, σ={NOISE}):")
    rows = {}
    for name in ["GraphRAG retrieval"] + list(models) + ["GCN (3-layer, trained)",
                                                         "Executor-Supervisor (twin-fit)"]:
        v = np.array(acc[name], dtype=float)
        rows[name] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                      "values": [float(x) for x in v]}   # per-seed runs for the figure dots
        print(f"   {name:34s}: {v.mean()*100:5.1f} ± {v.std(ddof=1)*100:4.1f}%")

    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seeds": SEEDS,
                          "protocol": "train on KB, test on system test leaks (forced)"},
           "forced_top1": rows}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_baselines_fair.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_baselines_fair.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
