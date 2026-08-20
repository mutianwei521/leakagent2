"""
Register-anchored SEMI-SYNTHETIC leak corpus for the City H leg.

The ONLY genuinely real City H dataset is the 2018-2020 municipal leak register
(field_validation_data/city_h_leak_statistics_2018_2020.csv): event metadata with
pipe diameter (管径), water-loss severity (漏失水量, tons/day), leak type (暗漏/明漏),
material (材质) and township DMA (所属计量区). Real SCADA pressure is confidential
(confidential) and unavailable, and the register carries NO field that keys to the network
model's numeric node IDs (verified: the 12 DMA names appear nowhere in city_h.inp;
addresses are free-text with no coordinates). The only honest, fully rule-based use
of this register is therefore DISTRIBUTION-anchored.

We build ONE synthetic leak per real register event (960), preserving the event's
REAL attributes:
  - leak rate  = 漏失水量(吨/天) / 86.4  (tons/day -> L/s; water 1 t = 1 m^3)
  - diameter, type, material, DMA        = carried verbatim as metadata
and place it on a network node by a documented, reproducible DIAMETER-MATCHED rule
(sample uniformly among junctions whose largest incident pipe diameter is closest to
the event's diameter; NO hand-picking). The leak's sensor pressure signature is then
forward-simulated in the nominal twin and field-grade Gaussian noise (sigma=0.15 m,
the documented transducer model) is added -- because no real pressure exists.

This is disclosed as semi-synthetic everywhere; external real-MEASUREMENT validity is
carried by the L-Town benchmark, NOT by this leg. Output: data/city_h/leak_corpus.json.
Resumable + time-guarded; per-event RNG is seeded by event id so results are
order-independent and fully reproducible.

Run (once, in the background):  python data/city_h_anchor_corpus.py
"""
import os
import sys
import json
import time
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context

REGISTER = r"C:\mutianwei\paper\dualLLMWDS\field_validation_data\city_h_leak_statistics_2018_2020.csv"
OUTJSON = os.path.join(ROOT, "data", "city_h", "leak_corpus.json")
SENSOR_STD = 0.15                  # field-grade transducer noise (m)
SEED = 42
TONS_PER_DAY_TO_LS = 1.0 / 86.4    # 1 ton/day water = 1 m^3/day = 1000 L / 86400 s
MAX_SEC = 5000
SAVE_EVERY = 100


def node_max_incident_diameter_mm(wn, n2c, junc):
    """For each JUNCTION (leaks can only be hosted on junctions, not the reservoir),
    the largest incident pipe diameter in mm (wntr stores m)."""
    jd = defaultdict(float)
    for ln in wn.pipe_name_list:
        l = wn.get_link(ln)
        d_mm = float(l.diameter) * 1000.0
        for nn in (str(l.start_node_name), str(l.end_node_name)):
            if nn in junc and nn in n2c:
                jd[nn] = max(jd[nn], d_mm)
    return jd


def save(corpus, sensors):
    out = {
        "provenance": {
            "network": "City H municipal WDN",
            "model": "nominal dataset/city_h.inp",
            "data": "SEMI-SYNTHETIC, register-anchored: real 2018-2020 leak register supplies "
                    "leak rate/diameter/type/material/DMA per event; node placement is a "
                    "documented diameter-matched rule; pressures are twin-simulated + Gaussian "
                    "field noise because real SCADA pressure is confidential/unavailable.",
            "mapping": "distribution_only (no register field keys to the model's node IDs)",
            "sensor_std_m": SENSOR_STD, "seed": SEED,
            "tons_per_day_to_Ls": "severity / 86.4",
        },
        "sensor_nodes": sensors,
        "n_leaks": len(corpus),
        "leaks": corpus,
    }
    json.dump(out, open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ctx = build_context(os.path.join(ROOT, "configs", "city_h.yaml"))
    wn = ctx.twin.wn
    n2c = ctx.node_to_partition
    sensors = ctx.sensors
    junc = set(str(x) for x in wn.junction_name_list)
    jd = node_max_incident_diameter_mm(wn, n2c, junc)
    nodes = sorted(jd.keys())
    node_diam = np.array([jd[n] for n in nodes], dtype=float)
    print(f"{len(nodes)} placeable junctions; {len(sensors)} sensors", flush=True)

    df = pd.read_csv(REGISTER, encoding="utf-8")
    print(f"register: {len(df)} real leak events", flush=True)

    existing = {}
    if os.path.exists(OUTJSON):
        try:
            existing = {int(e["id"]): e for e in json.load(open(OUTJSON, encoding="utf-8")).get("leaks", [])}
        except Exception:
            existing = {}
    corpus = list(existing.values())
    done_ids = set(existing)
    print(f"resuming: {len(done_ids)} already built", flush=True)

    t0 = time.time()
    newc = 0
    for _, row in df.iterrows():
        rid = int(row["序号"])
        if rid in done_ids:
            continue
        d_mm = float(row["管径"])
        sev = float(row["漏失水量(吨/天)"])
        rate = max(sev * TONS_PER_DAY_TO_LS, 1e-4)
        typ = str(row["漏点类型"])
        mat = str(row.get("材质", ""))
        dma = str(row["所属计量区"])

        # diameter-matched node pool: junctions whose largest incident pipe diameter
        # is within the nearest ~5% of |diff| to the event diameter (rule-based; no hand-pick)
        diff = np.abs(node_diam - d_mm)
        tol = max(float(np.percentile(diff, 5)), 1.0)
        pool = [nodes[j] for j in range(len(nodes)) if diff[j] <= tol] or nodes
        r2 = np.random.RandomState(SEED * 100003 + rid)
        node = pool[int(r2.randint(len(pool)))]

        dp = ctx.twin.leak_delta(node, rate, "day_normal")
        if dp is None:                       # non-physical solve (rare); skip honestly
            done_ids.add(rid)
            continue
        clean = {s: float(dp.get(s, 0.0)) for s in sensors}
        maxabs_clean = max((abs(v) for v in clean.values()), default=0.0)
        noise = r2.normal(0.0, SENSOR_STD, size=len(sensors))
        obs = {s: clean[s] + float(noise[k]) for k, s in enumerate(sensors)}
        tz = n2c.get(node)
        corpus.append({
            "id": rid, "leak_node": node,
            "true_zones": [int(tz)] if tz is not None else [],
            "rate_Ls": rate, "severity_tpd": sev, "diameter_mm": d_mm,
            "type": typ, "material": mat, "dma": dma,
            "max_abs_dp_clean_m": maxabs_clean, "obs_dp": obs,
        })
        done_ids.add(rid)
        newc += 1
        if newc % SAVE_EVERY == 0:
            save(corpus, sensors)
            print(f"   built {newc} new ({len(corpus)} total) {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > MAX_SEC:
            save(corpus, sensors)
            print(f"INCOMPLETE: {len(corpus)} built -- re-run to resume.", flush=True)
            return 2

    save(corpus, sensors)
    print(f"COMPLETE: {len(corpus)} synthetic leaks written -> data/city_h/leak_corpus.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
