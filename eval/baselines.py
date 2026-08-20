"""
Real, actually-trained leak-localization baselines on EXA7 (no fabrication).
Classical ML (k-NN / RandomForest / MLP / SVM) via 5-fold stratified CV, and a
genuinely-trained GCN (the prior repository hard-coded these numbers; here they
are computed). Reports forced (coverage=1) Top-1 and Macro-F1 at field noise.

Run:  python eval/baselines.py     (via the wds_rag interpreter)
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

DATA = os.path.join(ROOT, "data", "exa7")        # vendored, repo-relative
FP = os.path.join(DATA, "sensor_fingerprints.json")
TOPO = os.path.join(DATA, "topology_index.json")
ADJ = os.path.join(DATA, "partition_adjacency.json")
NOISE = 0.05
SEED = 42


def load_xy(noise=NOISE, seed=SEED):
    d = json.load(open(FP, encoding="utf-8"))
    sensors = [str(s) for s in d["sensor_nodes"]]
    rng = np.random.default_rng(seed)
    X, y, rates = [], [], []
    for f in d["fingerprints"]:
        v = np.asarray(f["sensor_fingerprint"], dtype=float)
        v = v + rng.normal(0, noise, size=v.shape)
        X.append(v); y.append(int(f["leak_partition"])); rates.append(float(f["leak_rate_Ls"]))
    return np.array(X), np.array(y), np.array(rates), sensors


def macro_f1(y, yp):
    from sklearn.metrics import f1_score
    return f1_score(y, yp, average="macro", zero_division=0)


def ml_baselines(X, y):
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "kNN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine"),
        "RandomForest (200)": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "MLP (128,64)": make_pipeline(StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                                    early_stopping=True, random_state=SEED)),
        "SVM (RBF, C=10)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED)),
    }
    out = {}
    for name, clf in models.items():
        yp = cross_val_predict(clf, X, y, cv=cv)
        out[name] = {"top1": float((yp == y).mean()), "macro_f1": float(macro_f1(y, yp))}
        print(f"   {name:22s}: Top-1 {out[name]['top1']*100:5.1f}%  Macro-F1 {out[name]['macro_f1']*100:5.1f}%")
    return out


def gcn_baseline(X, y, sensors):
    """A genuinely-trained GCN over per-scenario sensor graphs, 5-fold CV."""
    import torch
    import torch.nn.functional as Fnn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.model_selection import StratifiedKFold
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sensor graph edges: connect sensors in same or adjacent partitions
    topo = json.load(open(TOPO, encoding="utf-8"))
    n2p = {str(k): int(v) for k, v in topo.get("node_to_partition", {}).items()}
    adjd = json.load(open(ADJ, encoding="utf-8"))
    adj = adjd.get("adjacency", adjd)
    nbr = {int(k): set(int(x) for x in v) for k, v in adj.items()} if isinstance(adj, dict) else {}
    sp = [n2p.get(s, -1) for s in sensors]
    ei = []
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            if i == j:
                continue
            if sp[i] == sp[j] or sp[j] in nbr.get(sp[i], set()):
                ei.append([i, j])
    if not ei:
        ei = [[i, j] for i in range(len(sensors)) for j in range(len(sensors)) if i != j]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()

    classes = sorted(set(int(v) for v in y))
    cidx = {c: i for i, c in enumerate(classes)}
    graphs = [Data(x=torch.tensor(X[k], dtype=torch.float).view(-1, 1), edge_index=edge_index,
                   y=torch.tensor([cidx[int(y[k])]], dtype=torch.long)) for k in range(len(y))]

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y):
        model = GCN().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        trl = DataLoader([graphs[i] for i in tr], batch_size=32, shuffle=True)
        for ep in range(150):
            model.train()
            for b in trl:
                b = b.to(dev); opt.zero_grad()
                loss = Fnn.cross_entropy(model(b), b.y); loss.backward(); opt.step()
        model.eval()
        tel = DataLoader([graphs[i] for i in te], batch_size=64)
        p = []
        with torch.no_grad():
            for b in tel:
                p.append(model(b.to(dev)).argmax(1).cpu().numpy())
        preds[te] = np.concatenate(p)
    yp = np.array([classes[p] for p in preds])
    res = {"top1": float((yp == y).mean()), "macro_f1": float(macro_f1(y, yp))}
    print(f"   {'GCN (3-layer, trained)':22s}: Top-1 {res['top1']*100:5.1f}%  Macro-F1 {res['macro_f1']*100:5.1f}%")
    return res


def main():
    print("=" * 66)
    print(f"REAL TRAINED BASELINES on EXA7 (5-fold CV, σ={NOISE} m, seed={SEED})")
    print("=" * 66)
    X, y, rates, sensors = load_xy()
    print(f"dataset: {X.shape[0]} scenarios x {X.shape[1]} sensors, {len(set(y))} zones\n[ML]")
    res = ml_baselines(X, y)
    print("[GNN]")
    try:
        res["GCN (3-layer, trained)"] = gcn_baseline(X, y, sensors)
    except Exception as e:
        print("   GCN failed:", repr(e)[:160])
    best = max(res.items(), key=lambda kv: kv[1]["top1"])
    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seed": SEED, "cv": "5-fold stratified",
                          "note": "REAL trained baselines; in-distribution KB scenarios"},
           "baselines": res, "best": {"name": best[0], **best[1]}}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_baselines.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"\nbest forced localizer: {best[0]} Top-1 {best[1]['top1']*100:.1f}%  "
          f"(vs system selective decision-precision 97.6%)")
    print("saved -> artifacts/results_baselines.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
