# -*- coding: utf-8 -*-
"""
Survey tier, wave 2: CUSUM sequential detection versus the fixed-window mean.

The committed survey tier averages W nightly zone-inflow readings and applies a
single corrected threshold. CUSUM is the standard sequential alternative for
persistent shifts (night-flow practice): per zone, S_t = max(0, S_{t-1} + x_t - k)
with reference k, dispatch to the argmax zone when its S_t crosses h within the
event's real work-order window. h is calibrated per (sigma_f, k) by Monte Carlo
on no-leak campaigns so that the campaign-level false-alarm rate matches the
corrected fixed-window rule (~0.5% per campaign), then verified on the same 50
control campaigns used throughout.

Signals: the exact twin-computed per-zone inflow deltas (mass-balance verified).
One seeded noise realization per event (seed 42), as in the committed tier.

Run:  <wds_rag python> eval/run_city_d_flow_cusum.py   (pure numpy, no EPANET)
Output: artifacts/results_city_d_flow_cusum.json
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
W_CAP, N_CTRL, W_CTRL, SEED = 14, 50, 5, 42
TARGET_FA = 0.005          # per-campaign false-alarm target (matches u=3.4 rule)
N_CAL = 4000               # calibration campaigns per setting


def cusum_hit(x, k, h):
    """Return True if the one-sided CUSUM of series x crosses h."""
    s = 0.0
    for v in x:
        s = max(0.0, s + v - k)
        if s >= h:
            return True
    return False


def campaign_max_stat(noise, k):
    """Max over zones of the max CUSUM statistic within the window."""
    best = 0.0
    for z in range(noise.shape[0]):
        s = 0.0
        for v in noise[z]:
            s = max(0.0, s + v - k)
            best = max(best, s)
    return best


def main():
    src = json.load(open(os.path.join(ROOT, "artifacts", "results_city_d_ceiling_flow.json"),
                         encoding="utf-8"))
    events = src["events"]
    nz = len(events[0]["flow_delta_by_zone"])
    print("=" * 70)
    print(f"SURVEY TIER: CUSUM vs fixed-window mean ({len(events)} events, {nz} zones)")
    print("=" * 70)

    out = {"provenance": {"signal_source": "results_city_d_ceiling_flow.json (twin-computed deltas)",
                          "rule": "one-sided CUSUM per zone, k = sigma_f/2, h calibrated by "
                                  f"{N_CAL}-campaign Monte Carlo to ~{TARGET_FA*100:.1f}% campaign FA "
                                  "(matching the corrected fixed-window rule); dispatch to the argmax "
                                  "zone on first crossing within the real work-order window (cap 14)",
                          "seed": SEED, "n_controls": N_CTRL, "w_controls": W_CTRL},
           "results": {}}

    for sf in SIGMA_F_GRID:
        k = sf / 2.0
        # calibrate h on W_CTRL-night no-leak campaigns
        rng_cal = np.random.default_rng(SEED + 100)
        stats = np.array([campaign_max_stat(rng_cal.normal(0, sf, (nz, W_CTRL)), k)
                          for _ in range(N_CAL)])
        h = float(np.quantile(stats, 1.0 - TARGET_FA))
        # evaluate events
        rng = np.random.default_rng(SEED)
        acted = correct = 0
        for ev in events:
            W = max(1, min(ev["window_days"], W_CAP))
            zids = sorted(ev["flow_delta_by_zone"], key=int)
            sig = np.array([ev["flow_delta_by_zone"][z] for z in zids])
            x = sig[:, None] + rng.normal(0, sf, (len(zids), W))
            # first zone to cross h wins; approximate by max CUSUM peak per zone
            peaks = []
            for zi in range(len(zids)):
                s, pk = 0.0, 0.0
                for v in x[zi]:
                    s = max(0.0, s + v - k)
                    pk = max(pk, s)
                peaks.append(pk)
            j = int(np.argmax(peaks))
            if peaks[j] >= h:
                acted += 1
                if int(zids[j]) in set(ev["true_zones"]):
                    correct += 1
        # verify on the standard 50 control campaigns
        rngc = np.random.default_rng(SEED + 1)
        fa = sum(1 for _ in range(N_CTRL)
                 if campaign_max_stat(rngc.normal(0, sf, (nz, W_CTRL)), k) >= h)
        key = f"sigma{sf}"
        out["results"][key] = {"sigma_f_Ls": sf, "k": k, "h": round(h, 4),
                               "survey_dispatched": acted, "coverage": acted / len(events),
                               "zone_correct": correct,
                               "precision_on_dispatched": (correct / acted) if acted else None,
                               "controls_false_alarm": f"{fa}/{N_CTRL}"}
        print(f"  sigma_f={sf:.2f}: h={h:.3f}  dispatched {acted}/{len(events)} "
              f"({acted/len(events)*100:.0f}%), zone-correct {correct}/{acted or 1} "
              f"({(correct/acted*100 if acted else 0):.0f}%), controls FA {fa}/{N_CTRL}")

    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_flow_cusum.json"),
                        "w", encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_city_d_flow_cusum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
