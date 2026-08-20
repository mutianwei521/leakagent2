# -*- coding: utf-8 -*-
"""Stage 0 for the City D FIELD leg: vendor the utility INP and extract the
audited leak corpus from the mapping workbook (审计版) into data/city_d/.

Only records with 纳入主场景 = 是 are extracted (the workbook's strict filter:
leak volume present + acceptable connectivity + non-proxy anchor + node stable
across the two candidate CRS). Every source file's SHA-256 is recorded so the
corpus is traceable to the exact utility deliverables. Nothing is invented:
node assignments, rates and confidence flags are copied verbatim.

Run:  python data/city_d_extract_corpus.py   (via the wds_rag interpreter)
"""
import os
import sys
import json
import shutil
import hashlib

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.abspath(os.path.join(ROOT, "..", "realwds"))
OUT = os.path.join(ROOT, "data", "city_d")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "inp": os.path.join(SRC, "city_d.inp"),
        "register_xls": os.path.join(SRC, "utility_2025_leak_register_restricted.xls"),
        "mapping_xlsx": os.path.join(SRC, "city_d_leak_node_mapping_workbook_restricted.xlsx"),
    }
    prov = {k: {"file": os.path.basename(p), "sha256": sha256(p)} for k, p in files.items()}
    shutil.copyfile(files["inp"], os.path.join(OUT, "city_d.inp"))
    print("vendored city_d.inp")

    mp = pd.ExcelFile(files["mapping_xlsx"]).parse("漏点节点映射")
    main_rows = mp[mp["纳入主场景"] == "是"].copy()
    print(f"main-scenario records: {len(main_rows)} of {len(mp)}")

    corpus = []
    for _, r in main_rows.iterrows():
        corpus.append({
            "id": int(r["ID"]),
            "node": str(int(r["推荐仿真节点"])),
            "rate_Ls": float(r["需水增量L/s(公式)"]),
            "loss_t_per_day": float(r["漏量t/d"]),
            "diameter_mm": (None if pd.isna(r["原记录管径mm"]) else float(r["原记录管径mm"])),
            "material": (None if pd.isna(r["材质"]) else str(r["材质"])),
            "dma": (None if pd.isna(r["计量区域"]) else str(r["计量区域"])),
            "leak_type": (None if pd.isna(r["漏损类型"]) else str(r["漏损类型"])),
            "location_type": (None if pd.isna(r["位置类型"]) else str(r["位置类型"])),
            "report_date": str(r["上报日期"])[:10],
            "completion_date": str(r["完成日期"])[:10],
            "window_days": int(r["工单窗口天数"]),
            "anchor_confidence": str(r["地理置信"]),
            "merge_distance_m": (None if pd.isna(r["并入主干距离m"]) else float(r["并入主干距离m"])),
        })

    out = {
        "provenance": {
            "network": "city_d (City D district, utility EPANET model, title date withheld)",
            "register": "2025 field leak work orders (256 orders; audited mapping workbook)",
            "filter": "纳入主场景 = 是 (volume present + connectivity acceptable + non-proxy anchor + node stable across CRS candidates)",
            "sources": prov,
            "explicit_limits": [
                "no SCADA pressures or flows exist for these events: sensor responses are twin-simulated",
                "model vintage 2016 vs 2025 leaks: no 2025 recalibration is claimed",
                "coordinate datum unknown (provisional projected CRS); node mapping cross-checked across CRS candidates",
                "report-to-completion window is a work-order proxy, not a measured leak duration",
                "177/255 loss volumes are estimation tiers (multiples of 2.4 t/d)",
            ],
        },
        "n_records": len(corpus),
        "n_unique_nodes": len({c["node"] for c in corpus}),
        "records": corpus,
    }
    path = os.path.join(OUT, "leak_corpus.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(corpus)} records on {out['n_unique_nodes']} nodes")
    rates = [c["rate_Ls"] for c in corpus]
    rates.sort()
    print(f"rate L/s: median {rates[len(rates)//2]:.4f}, max {max(rates):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
