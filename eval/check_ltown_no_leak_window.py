# -*- coding: utf-8 -*-
"""
L-Town: is there any labelled no-leak window from which real no-leak controls
could be drawn?

Reads the official BattLeDIM dataset_configuration.yaml leak table (33 labelled
leaks with start/end times) and computes, over the benchmark span, how many
calendar days are intersected by at least one labelled leak interval and the
maximum number of simultaneously active leaks. This backs the Table 1 caption's
"n/a is a data fact" statement with committed numbers.

Run:  <wds_rag python> eval/check_ltown_no_leak_window.py   (pure python)
Output: artifacts/results_ltown_timeline.json
"""
import os
import sys
import json
import datetime as dt

import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT = os.environ.get("LTOWN_SCADA_DIR", r"data/ltown/scada")  # same source as eval/run_ltown.py
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    cfg = yaml.safe_load(open(os.path.join(LT, "dataset_configuration.yaml"), encoding="utf-8"))
    iv = []
    for item in cfg.get("leakages", []):
        if not isinstance(item, str) or item.strip().startswith("#"):
            continue
        p = [x.strip() for x in item.split(",")]
        if len(p) < 6:
            continue
        iv.append((pd.Timestamp(p[1]).to_pydatetime(), pd.Timestamp(p[2]).to_pydatetime()))
    t0, t1 = min(s for s, _ in iv), max(e for _, e in iv)

    day, free_days, total_days = t0.date(), 0, 0
    while day <= t1.date():
        total_days += 1
        d0 = dt.datetime.combine(day, dt.time(0))
        d1 = d0 + dt.timedelta(days=1)
        if not any(s < d1 and e > d0 for s, e in iv):     # interval intersects the day
            free_days += 1
        day += dt.timedelta(days=1)

    events = sorted([(s, 1) for s, _ in iv] + [(e, -1) for _, e in iv])
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)

    out = {"provenance": {"source": "BattLeDIM dataset_configuration.yaml (official leak labels)",
                          "definition": "a day is leak-free only if NO labelled leak interval "
                                        "intersects it; peak = max simultaneously active leaks"},
           "n_labelled_leaks": len(iv),
           "span": {"start": str(t0.date()), "end": str(t1.date()), "days": total_days},
           "days_with_at_least_one_active_leak": total_days - free_days,
           "leak_free_days": free_days,
           "max_simultaneous_leaks": peak}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_ltown_timeline.json"),
                        "w", encoding="utf-8"), indent=2)
    print(f"{len(iv)} labelled leaks | span {t0.date()}..{t1.date()} ({total_days} days) | "
          f"leak-free days: {free_days} | busy days: {total_days-free_days} | "
          f"max simultaneous: {peak}")
    print("saved -> artifacts/results_ltown_timeline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
