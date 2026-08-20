# -*- coding: utf-8 -*-
"""
Extend the City D twin leak cache to the register-leg hypothesis rate grid.

The executor's leak hypotheses are fitted over RATE_GRID (run_city_d.py), which
now extends down to the register's severity range (0.05-1.0 L/s added to the
standard 2-50 grid). This pre-simulates the added (candidate node, rate) deltas
so the register evaluation runs from cache; without this, every miss would
trigger a live EPANET solve inside the evaluation loop.

Run:  <wds_rag python> eval/extend_city_d_cache.py   (EPANET, ~1875 sims)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from eval.run_city_d import RATE_GRID, LEAK_CANDS

CACHE = os.path.join(ROOT, "data", "city_d", "leak_cache.pkl")
EXISTING_RATES = {2.0, 5.0, 10.0, 20.0, 35.0, 50.0}


def main():
    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    n0 = ctx.twin.load_cache(CACHE)
    new_rates = sorted(set(RATE_GRID) - EXISTING_RATES)
    cand = []
    for z in ctx.partitions:
        cand += ctx.top_degree_nodes(z, LEAK_CANDS)
    cand = list(dict.fromkeys(cand))
    total = len(cand) * len(new_rates)
    print(f"cache {n0} entries | extending: {len(cand)} nodes x rates {new_rates} "
          f"= {total} sims", flush=True)

    t0, done, fails = time.time(), 0, 0
    for r in new_rates:
        for node in cand:
            dp = ctx.twin.leak_delta(str(node), float(r), "day_normal")
            done += 1
            fails += dp is None
            if done % 200 == 0:
                ctx.twin.save_cache(CACHE)
                print(f"  {done}/{total} ({time.time()-t0:.0f}s, fails {fails})", flush=True)
    ctx.twin.save_cache(CACHE)
    print(f"DONE: {done} sims, {fails} non-physical, cache now "
          f"{ctx.twin.load_cache(CACHE)} entries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
