# -*- coding: utf-8 -*-
"""
FIELD scenario corpus for the City D leg.

Unlike the City H leg (whose register has no key to the model's node IDs, so
placement is a diameter-matched distribution rule), every City D record carries an
AUDITED real-location node assignment: the utility work-order address was geocoded
(internet anchor, three candidate CRS cross-checked) and merged to the nearest
numeric junction of the utility's own EPANET model, with per-record confidence and
merge distance preserved (data/city_d/leak_corpus.json, built from the audit
workbook). This script therefore does NO placement of its own: node and rate are
copied verbatim from the audited mapping.

What remains simulated, and why: no SCADA pressures or flows exist for these 2025
events, so each event's sensor signature is forward-simulated on the utility model
and field-grade Gaussian noise (sigma = 0.15 m) is added. Per-event RNG is seeded
by event id, so the corpus is order-independent and bit-reproducible.

Output: data/city_d/scenario_corpus.json.
Run (after setup + library):  python data/city_d_anchor_corpus.py
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context

INJSON = os.path.join(ROOT, "data", "city_d", "leak_corpus.json")
OUTJSON = os.path.join(ROOT, "data", "city_d", "scenario_corpus.json")
SENSOR_STD = 0.15
SEED = 42
MAX_SEC = 5000
SAVE_EVERY = 50


def save(corpus, sensors, src_prov):
    out = {
        "provenance": {
            "network": "City D district WDN (utility EPANET model, title date withheld)",
            "data": "FIELD register, audited mapping: real 2025 work orders supply rate/diameter/"
                    "type/DMA per event AND the audited real-location node assignment; pressures "
                    "are twin-simulated + Gaussian field noise (sigma=0.15 m) because no SCADA "
                    "exists for these events.",
            "mapping": "audited real-location (geocoded address -> nearest numeric junction; "
                       "per-record confidence and merge distance preserved) -- NOT distribution-placed",
            "sensor_std_m": SENSOR_STD, "seed": SEED,
            "source_files": src_prov,
        },
        "sensor_nodes": sensors,
        "n_leaks": len(corpus),
        "leaks": corpus,
    }
    json.dump(out, open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    n2c = ctx.node_to_partition
    sensors = ctx.sensors
    ctx.twin.load_cache(os.path.join(ROOT, "data", "city_d", "leak_cache.pkl"))

    src = json.load(open(INJSON, encoding="utf-8"))
    records = src["records"]
    src_prov = src["provenance"]["sources"]
    print(f"audited field records: {len(records)}; {len(sensors)} sensors", flush=True)

    existing = {}
    if os.path.exists(OUTJSON):
        try:
            prev = json.load(open(OUTJSON, encoding="utf-8"))
            prev_sensors = set(map(str, prev.get("provenance", {}).get("sensor_nodes", [])))
            if prev_sensors == set(map(str, sensors)):
                existing = {int(e["id"]): e for e in prev.get("leaks", [])}
            else:
                # RESUME TRAP: keeping entries built under a different sensor set
                # would mix obs_dp key spaces (this poisoned the register leg once).
                print(f"existing corpus was built under a DIFFERENT sensor set "
                      f"({len(prev_sensors & set(map(str, sensors)))}/{len(sensors)} shared) "
                      "- discarding it and rebuilding from scratch.", flush=True)
        except Exception:
            existing = {}
    corpus = list(existing.values())
    done = set(existing)
    print(f"resuming: {len(done)} already built", flush=True)

    t0 = time.time()
    newc, skipped = 0, []
    for rec in records:
        rid = int(rec["id"])
        if rid in done:
            continue
        node = str(rec["node"])
        rate = max(float(rec["rate_Ls"]), 1e-4)
        dp = ctx.twin.leak_delta(node, rate, "day_normal")
        if dp is None:                       # non-physical solve; record honestly and skip
            skipped.append(rid)
            done.add(rid)
            continue
        clean = {s: float(dp.get(s, 0.0)) for s in sensors}
        maxabs_clean = max((abs(v) for v in clean.values()), default=0.0)
        r2 = np.random.RandomState(SEED * 100003 + rid)
        noise = r2.normal(0.0, SENSOR_STD, size=len(sensors))
        obs = {s: clean[s] + float(noise[k]) for k, s in enumerate(sensors)}
        tz = n2c.get(node)
        corpus.append({
            "id": rid, "leak_node": node,
            "true_zones": [int(tz)] if tz is not None else [],
            "rate_Ls": rate, "severity_tpd": float(rec["loss_t_per_day"]),
            "diameter_mm": rec["diameter_mm"], "type": rec["leak_type"],
            "material": rec["material"], "dma": rec["dma"],
            "anchor_confidence": rec["anchor_confidence"],
            "merge_distance_m": rec["merge_distance_m"],
            "max_abs_dp_clean_m": maxabs_clean, "obs_dp": obs,
        })
        done.add(rid)
        newc += 1
        if newc % SAVE_EVERY == 0:
            save(corpus, sensors, src_prov)
            print(f"   built {newc} new ({len(corpus)} total) {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > MAX_SEC:
            save(corpus, sensors, src_prov)
            print(f"INCOMPLETE: {len(corpus)} built -- re-run to resume.", flush=True)
            return 2

    save(corpus, sensors, src_prov)
    # persist the exact-rate deltas so future integrity audits can replay the
    # projection without re-simulating
    ctx.twin.save_cache(os.path.join(ROOT, "data", "city_d", "leak_cache.pkl"))
    print(f"COMPLETE: {len(corpus)} field scenarios -> data/city_d/scenario_corpus.json "
          f"(skipped non-physical: {skipped or 'none'}; twin cache saved)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
