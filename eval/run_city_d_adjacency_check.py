# -*- coding: utf-8 -*-
"""
City D register: district-adjacency tolerance check (robustness of the register outcome).

Scores the register leg's FORCED top-1 district predictions (results_city_d.json
rows) as correct if the prediction lands in the true district OR any district
adjacent to it (data/city_d/partition_adjacency.json). The chance rate is the
mean over districts of |{z} + neighbours(z)| / 15, the definition used when the
check was first computed inline. Appends the block to
artifacts/results_city_d_flowtier.json (idempotent), which Table S15 and the
Results text cite; this formalizes what was previously an inline computation.

Run:  <wds_rag python> eval/run_city_d_adjacency_check.py   (pure python, no EPANET)
"""
import os
import sys
import json
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    zadj = json.load(open(os.path.join(ROOT, "data", "city_d", "partition_adjacency.json"),
                          encoding="utf-8"))
    zadj = zadj.get("adjacency", zadj)
    adj = {int(k): set(int(x) for x in v) for k, v in zadj.items()}
    n_zones = 15

    rows = json.load(open(os.path.join(ROOT, "artifacts", "results_city_d.json"),
                          encoding="utf-8"))["rows"]
    # FAIL LOUD if the artifact predates the genuinely-forced fields: scoring a
    # None prediction as a miss silently turned 194 absent predictions into
    # 194 misses once.
    if any("retrieval_top1" not in r or r["retrieval_top1"] is None for r in rows):
        raise RuntimeError("rows lack a real forced prediction (retrieval_top1); "
                           "re-run eval/run_city_d.py first.")

    def score(field):
        strict = loose = 0
        for r in rows:
            true_z = set(int(z) for z in r["true_zones"])
            ok = true_z | set().union(*(adj.get(z, set()) for z in true_z))
            p = r[field]
            if p is None:
                continue
            strict += int(p) in true_z
            loose += int(p) in ok
        return strict, loose

    chance = statistics.mean(len({z} | adj.get(z, set())) for z in adj) / n_zones
    n = len(rows)
    ret_s, ret_l = score("retrieval_top1")
    twf_s, twf_l = score("twinfit_top1")
    print(f"retrieval forced top-1: strict {ret_s}/{n}, with adjacency {ret_l}/{n} "
          f"(chance {100/n_zones:.1f}% / ~{chance*100:.1f}%)")
    print(f"twin-fit  forced top-1: strict {twf_s}/{n}, with adjacency {twf_l}/{n}")

    fp = os.path.join(ROOT, "artifacts", "results_city_d_flowtier.json")
    art = json.load(open(fp, encoding="utf-8"))
    art["adjacency_tolerance_check"] = {
        "definition": "genuinely forced zone predictions scored correct if they land in the "
                      "true district OR any adjacent district",
        "retrieval_top1": {"strict": f"{ret_s}/{n}", "with_adjacency": f"{ret_l}/{n}"},
        "twinfit_top1": {"strict": f"{twf_s}/{n}", "with_adjacency": f"{twf_l}/{n}"},
        "chance_rate_strict": round(1.0 / n_zones, 3),
        "chance_rate_adjacency_approx": round(chance, 3),
        "script": "eval/run_city_d_adjacency_check.py"}
    json.dump(art, open(fp, "w", encoding="utf-8"), indent=2)
    print(f"block written into {os.path.basename(fp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
