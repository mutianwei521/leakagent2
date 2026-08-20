"""
Resumable, time-guarded builder for the City H twin leak-RESPONSE library
(the per-candidate-node x rate Δp the executor fits observations against).
Persists to data/city_h/leak_cache.pkl and SKIPS already-computed entries, so it
survives interruption: re-run until it prints COMPLETE. Mirrors
data/ltown_build_library.py exactly (same LEAK_CANDS/RATES) for cross-leg parity.

Run repeatedly (or once in the background):  python data/city_h_build_library.py
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

CACHE = os.path.join(ROOT, "data", "city_h", "leak_cache.pkl")
LEAK_CANDS = 25
RATES = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
MAX_SEC = 5000         # generous; intended to run in the background to completion
SAVE_EVERY = 200


def main():
    ctx = build_context(os.path.join(ROOT, "configs", "city_h.yaml"))
    twin = ctx.twin
    have = twin.load_cache(CACHE)
    cand = []
    for z in ctx.partitions:
        cand += ctx.top_degree_nodes(z, LEAK_CANDS)
    cand = list(dict.fromkeys(cand))
    jobs = [(n, r) for n in cand for r in RATES]
    total = len(jobs)
    print(f"library jobs: {total}  (already cached: {have})", flush=True)

    t0 = time.time()
    done_now = 0
    for n, r in jobs:
        ck = f"leak|{n}|{float(r)}|day_normal"
        if ck in twin._delta_cache:
            continue
        twin.leak_delta(n, r, "day_normal")
        done_now += 1
        if done_now % SAVE_EVERY == 0:
            twin.save_cache(CACHE)
            print(f"   computed {done_now} new (cache={len(twin._delta_cache)}) "
                  f"{time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > MAX_SEC:
            twin.save_cache(CACHE)
            print(f"INCOMPLETE: cache={len(twin._delta_cache)} -- re-run to resume.", flush=True)
            return 2
    twin.save_cache(CACHE)
    miss = sum(1 for (n, r) in jobs if f"leak|{n}|{float(r)}|day_normal" not in twin._delta_cache)
    print(f"COMPLETE: cache={len(twin._delta_cache)} entries, {miss} unreproducible.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
