# -*- coding: utf-8 -*-
"""
City D register: pressure-information ceiling + zone flow-balance signal.

One simulation pass over the 194 audited register events computes, per event:

  1. CEILING: the maximum clean |dp| over ALL 541 junctions (a sensor at every
     node). If even this cannot clear the noise/actionability floors, no sensor
     placement can, closing the pressure-only question rigorously.
  2. FLOW SIGNAL: the change in each zone's net inflow (sum of flows on
     zone-boundary links, oriented inward, plus any source/tank inside the
     zone), twin-computed rather than assumed. Mass balance predicts the leak's
     zone gains ~q; the simulation verifies it and captures redistribution.

A sizing evaluator then asks: with district flow metering of nightly
uncertainty sigma_f (grid 0.05/0.10/0.20 L/s) averaged over the event's real
work-order window (W = min(window_days, 14) nights), how many register events
clear a 3-sigma detection with the correct zone leading? This sizes the
survey-dispatch tier honestly before any pipeline integration.

Run:  <wds_rag python> eval/run_city_d_ceiling_flow.py
Output: artifacts/results_city_d_ceiling_flow.json
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
SIGMA_F_GRID = (0.05, 0.10, 0.20)   # nightly zone-inflow meter uncertainty, L/s
W_CAP = 14                           # cap on nights averaged (stationarity)
FLOORS = (0.15, 0.45)


def zone_map():
    parts = pickle.load(open(os.path.join(ROOT, "data", "city_d", "partitions.pkl"), "rb"))
    K = list(parts.keys())[0]
    return {str(n): int(c) for n, c in parts[K]["node_to_community"].items()}


def run_avg(wn):
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()
    pfull = res.node["pressure"]                 # time x node
    q = res.link["flowrate"].mean(axis=0)
    return pfull.mean(axis=0), q, pfull


def main():
    n2z_all = zone_map()
    wn0 = wntr.network.WaterNetworkModel(INP)
    junc = [str(j) for j in wn0.junction_name_list]
    # SOURCES ARE EXTERNAL TO ALL ZONES: partitions.pkl assigns reservoir 2506 to
    # zone 1, which made its feed pipes intra-zone and silently dropped the
    # source term from zone 1's net inflow (a leak supplied by the in-zone
    # reservoir produced ~zero boundary-link signal). Restrict the zone map to
    # junctions so source-feed links count as inflow into the receiving zone.
    srcs = set(map(str, wn0.reservoir_name_list)) | set(map(str, wn0.tank_name_list))
    n2z = {n: z for n, z in n2z_all.items() if n not in srcs}
    zones = sorted(set(n2z.values()))

    # zone-boundary links (oriented: flow start->end positive); includes links
    # with exactly one zoned endpoint (source feeds)
    blinks = []
    for ln in wn0.link_name_list:
        lk = wn0.get_link(ln)
        za, zb = n2z.get(str(lk.start_node_name)), n2z.get(str(lk.end_node_name))
        if za == zb:
            continue                              # internal (or fully external)
        blinks.append((ln, za, zb))
    n_src = sum(1 for _, za, zb in blinks if za is None or zb is None)
    print(f"zones: {len(zones)} | boundary links: {len(blinks)} (incl. {n_src} source feeds)")

    def zone_inflow(qser):
        infl = {z: 0.0 for z in zones}
        for ln, za, zb in blinks:
            f = float(qser[ln]) * 1000.0        # m3/s -> L/s
            if zb is not None:
                infl[zb] += f                    # into end zone
            if za is not None:
                infl[za] -= f                    # out of start zone
        return infl

    p0, q0, pfull0 = run_avg(wntr.network.WaterNetworkModel(INP))
    infl0 = zone_inflow(q0)

    corpus = json.load(open(os.path.join(ROOT, "data", "city_d", "scenario_corpus.json"),
                            encoding="utf-8"))["leaks"]
    # str keys: register ids are ints, lookups below use str(e["id"]). An int-keyed
    # dict silently missed all 194 lookups and defaulted every window to 1 night.
    reg = {str(r["id"]): r for r in json.load(open(os.path.join(ROOT, "data", "city_d",
                                               "leak_corpus.json"), encoding="utf-8"))["records"]}

    events = []
    for i, e in enumerate(corpus):
        node, rate = str(e["leak_node"]), float(e["rate_Ls"])
        wn = wntr.network.WaterNetworkModel(INP)
        wn.get_node(node).demand_timeseries_list.append((rate / 1000.0, None, "LEAK_X"))
        try:
            p1, q1, pfull1 = run_avg(wn)
        except Exception as ex:
            print(f"  SIM FAIL {e['id']}: {ex}")
            continue
        dp = (p1 - p0).abs()
        ceil = float(dp[junc].max())
        # strict any-time bound: max over junctions AND report timesteps (the
        # time-mean convention above matches the pipeline's observation model;
        # both are reported)
        ceil_any = float((pfull1[junc] - pfull0[junc]).abs().values.max())
        infl1 = zone_inflow(q1)
        dz = {z: infl1[z] - infl0[z] for z in zones}
        true_z = set(int(z) for z in e["true_zones"])
        best_z = max(dz, key=lambda z: dz[z])
        wdays = int(reg[str(e["id"])]["window_days"]) if str(e["id"]) in reg else 1
        events.append({"id": e["id"], "rate_Ls": rate, "true_zones": sorted(true_z),
                       "window_days": wdays,
                       "ceiling_max_dp_all_nodes_m": round(ceil, 5),
                       "ceiling_anytime_m": round(ceil_any, 5),
                       "flow_delta_true_zone_Ls": round(max(dz[z] for z in true_z), 5),
                       "flow_best_zone": int(best_z),
                       "flow_best_is_true": bool(best_z in true_z),
                       "flow_margin_Ls": round(sorted(dz.values())[-1] - sorted(dz.values())[-2], 5),
                       "flow_delta_by_zone": {str(z): round(v, 6) for z, v in dz.items()}})
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(corpus)}", flush=True)
        for f in ("temp.bin", "temp.inp", "temp.rpt"):
            try:
                os.remove(os.path.join(ROOT, f))
            except OSError:
                pass

    ceils = np.array([ev["ceiling_max_dp_all_nodes_m"] for ev in events])
    ceils_any = np.array([ev["ceiling_anytime_m"] for ev in events])
    print("\n=== PRESSURE CEILING (sensor at every one of 541 junctions) ===")
    print(f"  time-mean convention: median {np.median(ceils):.4f} m | "
          f"p90 {np.percentile(ceils,90):.4f} | max {ceils.max():.4f}")
    print(f"  any-time bound:       median {np.median(ceils_any):.4f} m | "
          f"p90 {np.percentile(ceils_any,90):.4f} | max {ceils_any.max():.4f}")
    for fl in FLOORS:
        print(f"  events clearing {fl} m: time-mean {(ceils>=fl).sum()}/{len(events)} | "
              f"any-time {(ceils_any>=fl).sum()}/{len(events)}")

    print("\n=== FLOW-TIER SIZING (3-sigma detection, W = min(window,14) nights) ===")
    sizing = {}
    for sf in SIGMA_F_GRID:
        det = corr = 0
        for ev in events:
            W = max(1, min(ev["window_days"], W_CAP))
            thr = 3.0 * sf / np.sqrt(W)
            if ev["flow_delta_true_zone_Ls"] >= thr:
                det += 1
                if ev["flow_best_is_true"]:
                    corr += 1
        sizing[str(sf)] = {"detected": det, "coverage": det / len(events),
                           "zone_correct_among_detected": (corr / det) if det else None}
        print(f"  sigma_f={sf:.2f} L/s: detect {det}/{len(events)} "
              f"({det/len(events)*100:.0f}%), correct zone among detected "
              f"{(corr/det*100 if det else 0):.0f}%")

    flow_true = np.array([ev["flow_delta_true_zone_Ls"] for ev in events])
    rate = np.array([ev["rate_Ls"] for ev in events])
    print(f"\n  mass-balance check: median(flow_delta_true/rate) = "
          f"{np.median(flow_true/np.maximum(rate,1e-9)):.3f} (expect ~1)")
    print(f"  zone-ID correctness (argmax inflow delta, noiseless): "
          f"{sum(ev['flow_best_is_true'] for ev in events)}/{len(events)}")

    out = {"provenance": {"inp": "data/city_d/city_d.inp", "n_events": len(events),
                          "method": "direct WNTR sims, baseline vs per-event leak; ceiling = max "
                                    "clean |dp| over all junctions; flow = zone net-inflow delta "
                                    "via boundary-link flow sums (twin-computed, not assumed)",
                          "sigma_f_grid_Ls": SIGMA_F_GRID, "w_cap_nights": W_CAP},
           "ceiling": {"median_m": float(np.median(ceils)), "p90_m": float(np.percentile(ceils, 90)),
                       "max_m": float(ceils.max()),
                       "n_ge_0.15": int((ceils >= 0.15).sum()), "n_ge_0.45": int((ceils >= 0.45).sum())},
           "ceiling_anytime": {"median_m": float(np.median(ceils_any)),
                               "p90_m": float(np.percentile(ceils_any, 90)),
                               "max_m": float(ceils_any.max()),
                               "n_ge_0.15": int((ceils_any >= 0.15).sum()),
                               "n_ge_0.45": int((ceils_any >= 0.45).sum())},
           "flow_sizing": sizing,
           "zone_id_correct_noiseless": int(sum(ev["flow_best_is_true"] for ev in events)),
           "events": events}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_ceiling_flow.json"),
                        "w", encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_city_d_ceiling_flow.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
