"""
Resumable, time-guarded builder for the KY4 twin leak-response library.
Persists to data/ky4/leak_cache.pkl and SKIPS already-computed (node,rate)
entries, so it survives being killed: re-run until it prints COMPLETE.

Run repeatedly:  python data/ky4_build_library.py    (via wds_rag interpreter)
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

CACHE = os.path.join(ROOT, "data", "ky4", "leak_cache.pkl")
LEAK_CANDS = 25              # >= the executor's runtime leak_cands (25)
RATES = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
MAX_SEC = 520                # exit before the 600 s foreground limit, saving progress
SAVE_EVERY = 200


def main():
    ctx = build_context(os.path.join(ROOT, "configs", "ky4.yaml"))
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
    for k, (n, r) in enumerate(jobs):
        ck = f"leak|{n}|{float(r)}|day_normal"
        if ck in twin._delta_cache:
            continue
        twin.leak_delta(n, r, "day_normal")
        done_now += 1
        if done_now % SAVE_EVERY == 0:
            twin.save_cache(CACHE)
            print(f"   computed {done_now} new (cache={len(twin._delta_cache)}/{total}) "
                  f"{time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > MAX_SEC:
            twin.save_cache(CACHE)
            print(f"INCOMPLETE: cache={len(twin._delta_cache)}/{total} — re-run to resume.", flush=True)
            return 2
    twin.save_cache(CACHE)
    miss = sum(1 for (n, r) in jobs if f"leak|{n}|{float(r)}|day_normal" not in twin._delta_cache)
    print(f"COMPLETE: cache={len(twin._delta_cache)} entries, {miss} unreproducible.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
