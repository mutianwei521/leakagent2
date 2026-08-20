# -*- coding: utf-8 -*-
"""
City D register: demand-ratio sensitivity of the excavation ceiling and the
survey (flow) tier.

Question under test: could the field-leg performance be limited by an
inaccurate demand fit? The vendored model already carries the evidence-backed
2025 update (uniform ratio 1.168459 on the 2016 base, verified record-by-record
against the nine-year workbook). This program sweeps the overall demand ratio r
across the workbook's own hydraulic-sensitivity band (0.8 to 1.3 on the 2016
base) and recomputes, per ratio, exactly what run_city_d_ceiling_flow.py
computes on the committed model:

  1. CEILING: max clean |dp| over ALL 541 junctions per event (sensor at every
     node), with counts clearing the 0.15 m and 0.45 m floors.
  2. FLOW TIER: twin-computed zone net-inflow deltas, mass-balance ratio,
     noiseless zone-ID correctness, and detection counts under both the 3-sigma
     sizing rule and the committed corrected rule (u = 3.4), sigma_f = 0.10 L/s,
     W = min(window_days, 14).

The r = 1.168459 leg must reproduce the committed artifact (regression anchor).
Demands are scaled by f = r / 1.168459 on every junction demand record (zero
demands stay zero, patterns preserved); the per-event leak demand is appended
AFTER scaling and is never scaled. Nothing else changes.

This is a robustness/refutation study, not a tuning run: no ratio other than
the evidence-backed 1.168459 is eligible for adoption regardless of outcome.

Run:  <wds_rag python> eval/run_city_d_demand_sensitivity.py   (EPANET, ~1000 sims)
Output: artifacts/results_city_d_demand_sensitivity.json
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import wntr

INP = os.path.join(ROOT, "data", "city_d", "city_d.inp")
R_FORMAL = 1.168459                  # evidence-backed 2025 ratio, already in the INP
R_GRID = (0.8, 1.0, 1.1, R_FORMAL, 1.3)   # workbook hydraulic-sensitivity band
SIGMA_F = 0.10                       # committed survey-tier headline, L/s
U_CORR = 3.4                         # 15-district-corrected threshold multiplier
W_CAP = 14
FLOORS = (0.15, 0.45)


def zone_map():
    parts = pickle.load(open(os.path.join(ROOT, "data", "city_d", "partitions.pkl"), "rb"))
    K = list(parts.keys())[0]
    return {str(n): int(c) for n, c in parts[K]["node_to_community"].items()}


def scaled_model(f):
    """Fresh model with every junction demand record scaled by f (patterns kept)."""
    wn = wntr.network.WaterNetworkModel(INP)
    if abs(f - 1.0) > 1e-12:
        for jn in wn.junction_name_list:
            for d in wn.get_node(jn).demand_timeseries_list:
                d.base_value = float(d.base_value) * f
    return wn


def run_avg(wn):
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()
    p = res.node["pressure"].mean(axis=0)
    q = res.link["flowrate"].mean(axis=0)
    return p, q


def main():
    n2z_all = zone_map()
    wn0 = wntr.network.WaterNetworkModel(INP)
    junc = [str(j) for j in wn0.junction_name_list]
    total0 = sum(float(d.base_value) for jn in wn0.junction_name_list
                 for d in wn0.get_node(jn).demand_timeseries_list) * 1000.0

    # sources external to all zones (same accounting as run_city_d_ceiling_flow.py:
    # source-feed links count as inflow into the receiving zone)
    srcs = set(map(str, wn0.reservoir_name_list)) | set(map(str, wn0.tank_name_list))
    n2z = {n: z for n, z in n2z_all.items() if n not in srcs}
    zones = sorted(set(n2z.values()))

    blinks = []
    for ln in wn0.link_name_list:
        lk = wn0.get_link(ln)
        za, zb = n2z.get(str(lk.start_node_name)), n2z.get(str(lk.end_node_name))
        if za == zb:
            continue
        blinks.append((ln, za, zb))
    print(f"zones: {len(zones)} | boundary links: {len(blinks)} | "
          f"vendored base demand {total0:.1f} L/s (= 2016 base x {R_FORMAL})")

    def zone_inflow(qser):
        infl = {z: 0.0 for z in zones}
        for ln, za, zb in blinks:
            f = float(qser[ln]) * 1000.0
            if zb is not None:
                infl[zb] += f
            if za is not None:
                infl[za] -= f
        return infl

    corpus = json.load(open(os.path.join(ROOT, "data", "city_d", "scenario_corpus.json"),
                            encoding="utf-8"))["leaks"]
    # str keys to match the str(e["id"]) lookups (register ids are ints)
    reg = {str(r["id"]): r for r in json.load(open(os.path.join(ROOT, "data", "city_d",
                                               "leak_corpus.json"), encoding="utf-8"))["records"]}

    out = {"provenance": {"inp": "data/city_d/city_d.inp (demands = 2016 base x 1.168459)",
                          "method": "run_city_d_ceiling_flow.py replicated per ratio; demands "
                                    "scaled by r/1.168459 on every junction record before the "
                                    "leak demand is appended; leak rates never scaled",
                          "r_grid_on_2016_base": list(R_GRID), "sigma_f_Ls": SIGMA_F,
                          "u_corrected": U_CORR, "w_cap_nights": W_CAP,
                          "purpose": "robustness/refutation of the demand-staleness hypothesis; "
                                     "no ratio other than 1.168459 is eligible for adoption"},
           "ratios": {}}

    for r in R_GRID:
        f = r / R_FORMAL
        print(f"\n=== ratio r={r:g} on 2016 base (scale f={f:.6f} on vendored) ===", flush=True)
        p0, q0 = run_avg(scaled_model(f))
        infl0 = zone_inflow(q0)
        print(f"  baseline: min junction pressure {float(p0[junc].min()):.2f} m", flush=True)

        events = []
        for i, e in enumerate(corpus):
            node, rate = str(e["leak_node"]), float(e["rate_Ls"])
            wn = scaled_model(f)
            wn.get_node(node).demand_timeseries_list.append((rate / 1000.0, None, "LEAK_X"))
            try:
                p1, q1 = run_avg(wn)
            except Exception as ex:
                print(f"  SIM FAIL {e['id']}: {ex}")
                continue
            dp = (p1 - p0).abs()
            infl1 = zone_inflow(q1)
            dz = {z: infl1[z] - infl0[z] for z in zones}
            true_z = set(int(z) for z in e["true_zones"])
            best_z = max(dz, key=lambda z: dz[z])
            wdays = int(reg[str(e["id"])]["window_days"]) if str(e["id"]) in reg else 1
            events.append({"id": e["id"], "rate_Ls": rate,
                           "window_days": wdays,
                           "ceiling_max_dp_all_nodes_m": round(float(dp[junc].max()), 5),
                           "flow_delta_true_zone_Ls": round(max(dz[z] for z in true_z), 5),
                           "flow_best_is_true": bool(best_z in true_z)})
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(corpus)}", flush=True)
            for tf in ("temp.bin", "temp.inp", "temp.rpt"):
                try:
                    os.remove(os.path.join(ROOT, tf))
                except OSError:
                    pass

        ceils = np.array([ev["ceiling_max_dp_all_nodes_m"] for ev in events])
        ft = np.array([ev["flow_delta_true_zone_Ls"] for ev in events])
        rt = np.array([ev["rate_Ls"] for ev in events])
        det3 = detu = corr3 = corru = 0
        for ev in events:
            W = max(1, min(ev["window_days"], W_CAP))
            d = ev["flow_delta_true_zone_Ls"]
            if d >= 3.0 * SIGMA_F / np.sqrt(W):
                det3 += 1
                corr3 += ev["flow_best_is_true"]
            if d >= U_CORR * SIGMA_F / np.sqrt(W):
                detu += 1
                corru += ev["flow_best_is_true"]
        row = {"n_events": len(events),
               "baseline_min_pressure_m": round(float(p0[junc].min()), 3),
               "ceiling": {"median_m": round(float(np.median(ceils)), 5),
                           "p90_m": round(float(np.percentile(ceils, 90)), 5),
                           "max_m": round(float(ceils.max()), 5),
                           "n_ge_0.15": int((ceils >= FLOORS[0]).sum()),
                           "n_ge_0.45": int((ceils >= FLOORS[1]).sum())},
               "flow": {"massbal_median": round(float(np.median(ft / np.maximum(rt, 1e-9))), 4),
                        "zone_id_correct_noiseless": int(sum(ev["flow_best_is_true"]
                                                             for ev in events)),
                        "detect_3sigma": det3, "zone_correct_3sigma": corr3,
                        "detect_u3.4": detu, "zone_correct_u3.4": corru},
               "events": events}
        out["ratios"][str(r)] = row
        c = row["ceiling"]
        print(f"  CEILING: median {c['median_m']:.4f} m | p90 {c['p90_m']:.4f} | "
              f"max {c['max_m']:.4f} | >=0.15: {c['n_ge_0.15']}/{len(events)} | "
              f">=0.45: {c['n_ge_0.45']}/{len(events)}")
        fl = row["flow"]
        print(f"  FLOW: massbal {fl['massbal_median']:.3f} | zone-ID {fl['zone_id_correct_noiseless']}"
              f"/{len(events)} | detect(u=3.4,s=0.10) {fl['detect_u3.4']}/{len(events)}", flush=True)

    # regression anchor: r = 1.168459 must reproduce the committed artifact
    committed = json.load(open(os.path.join(ROOT, "artifacts",
                               "results_city_d_ceiling_flow.json"), encoding="utf-8"))
    anchor = out["ratios"][str(R_FORMAL)]
    checks = {
        "n_events": anchor["n_events"] == committed["provenance"]["n_events"],
        "ceiling_median": abs(anchor["ceiling"]["median_m"]
                              - committed["ceiling"]["median_m"]) < 1e-4,
        "n_ge_0.15": anchor["ceiling"]["n_ge_0.15"] == committed["ceiling"]["n_ge_0.15"],
        "n_ge_0.45": anchor["ceiling"]["n_ge_0.45"] == committed["ceiling"]["n_ge_0.45"],
        "detect_3sigma": anchor["flow"]["detect_3sigma"]
                         == committed["flow_sizing"]["0.1"]["detected"],
        "zone_id_correct": anchor["flow"]["zone_id_correct_noiseless"]
                           == committed["zone_id_correct_noiseless"],
    }
    out["regression_anchor_checks"] = {k: bool(v) for k, v in checks.items()}
    out["regression_anchor_ok"] = bool(all(checks.values()))
    print(f"\nregression anchor vs committed ceiling artifact: "
          + ", ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in checks.items()))

    json.dump(out, open(os.path.join(ROOT, "artifacts",
                        "results_city_d_demand_sensitivity.json"), "w", encoding="utf-8"),
              indent=2)
    print("saved -> artifacts/results_city_d_demand_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
