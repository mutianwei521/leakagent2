# -*- coding: utf-8 -*-
"""
City D survey-dispatch tier: realized-noise evaluation of the zone flow-balance
evidence, with no-leak controls.

Signal: per-event, per-zone net-inflow deltas twin-computed by
eval/run_city_d_ceiling_flow.py (mass-balance verified, median ratio 1.000).
Observation model: nightly zone-inflow estimates with meter uncertainty
sigma_f, averaged over the event's real work-order window (W = min(days, 14)
nights), so sigma_eff = sigma_f / sqrt(W). One seeded noise realization per
event (seed registry style, matching the pressure leg).

Decision rule (G2b, survey dispatch): let z* be the zone with the largest
observed inflow rise. Dispatch a survey crew to z* iff its rise clears u *
sigma_eff. Two thresholds are reported: u = 3 (plain 3-sigma) and u = 3.4
(15-zone Bonferroni-corrected, per-campaign alpha ~ 0.5%). Controls: 50
no-leak campaigns (pure noise, W = 5, the register median).

Run:  <wds_rag python> eval/run_city_d_flow_tier.py
Output: artifacts/results_city_d_flowtier.json
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIGMA_F_GRID = (0.05, 0.10, 0.20)
U_GRID = (3.0, 3.4)
W_CAP, N_CTRL, W_CTRL, SEED = 14, 50, 5, 42
N_ZONES = 15


def main():
    src = json.load(open(os.path.join(ROOT, "artifacts", "results_city_d_ceiling_flow.json"),
                         encoding="utf-8"))
    events = src["events"]
    print("=" * 70)
    print(f"CITY_D SURVEY TIER: realized noise, {len(events)} register events")
    print("=" * 70)

    out = {"provenance": {"signal_source": "results_city_d_ceiling_flow.json (twin-computed zone "
                                           "net-inflow deltas; mass-balance ratio 1.000)",
                          "obs_model": "nightly zone-inflow with meter sigma_f, averaged over "
                                       "W = min(window_days, 14) nights; one seeded realization "
                                       "per event (seed 42)",
                          "rule": "survey-dispatch the argmax zone iff its observed rise >= "
                                  "u * sigma_f/sqrt(W); u = 3.0 and 3.4 (15-zone corrected)",
                          "controls": f"{N_CTRL} no-leak campaigns, W = {W_CTRL} nights (register median)",
                          "sigma_f_grid_Ls": SIGMA_F_GRID, "w_cap": W_CAP},
           "results": {}}

    for sf in SIGMA_F_GRID:
        for u in U_GRID:
            rng = np.random.default_rng(SEED)
            acted = correct = 0
            for ev in events:
                W = max(1, min(ev["window_days"], W_CAP))
                se = sf / np.sqrt(W)
                zids = sorted(ev["flow_delta_by_zone"], key=int)
                sig = np.array([ev["flow_delta_by_zone"][z] for z in zids])
                obs = sig + rng.normal(0.0, se, len(zids))
                j = int(np.argmax(obs))
                if obs[j] >= u * se:
                    acted += 1
                    if int(zids[j]) in set(ev["true_zones"]):
                        correct += 1
            # controls
            rngc = np.random.default_rng(SEED + 1)
            se_c = sf / np.sqrt(W_CTRL)
            fa = 0
            for _ in range(N_CTRL):
                obs = rngc.normal(0.0, se_c, N_ZONES)
                if obs.max() >= u * se_c:
                    fa += 1
            key = f"sigma{sf}_u{u}"
            out["results"][key] = {
                "sigma_f_Ls": sf, "u": u, "n_events": len(events),
                "survey_dispatched": acted, "coverage": acted / len(events),
                "zone_correct": correct,
                "precision_on_dispatched": (correct / acted) if acted else None,
                "controls_false_alarm": f"{fa}/{N_CTRL}", "fa_rate": fa / N_CTRL}
            print(f"  sigma_f={sf:.2f} u={u}: dispatched {acted}/{len(events)} "
                  f"({acted/len(events)*100:.0f}%), zone-correct {correct}/{acted or 1} "
                  f"({(correct/acted*100 if acted else 0):.0f}%), controls FA {fa}/{N_CTRL}")

    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_flowtier.json"),
                        "w", encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_city_d_flowtier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
