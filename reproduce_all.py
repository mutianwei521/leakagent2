# -*- coding: utf-8 -*-
"""One-command reproduction driver for the executor-supervisor study.

Runs every step that produces a reported number or figure, in dependency order.
Physics/statistics steps are deterministic (fixed seed registry) and need no
network. The language-model steps need a running ollama runtime and are OPT-IN.

Usage (via the wds_rag interpreter):
  python reproduce_all.py               # smoke tests + every physics/stat step + figures + SI
  python reproduce_all.py --quick       # smoke tests + figures + SI only (fast sanity)
  python reproduce_all.py --with-llm    # also re-run the three language-model experiments
  python reproduce_all.py --only exa7   # one group: exa7 | transfer | real | battery | figs | llm

Groups and what they need:
  exa7      data/exa7 (vendored)                       EXA7 headline, pooled stats, baselines,
                                                       calibration, threshold sweep, ablations
  transfer  data/ky4 (vendored), data/city_h, data/city_d (from the corresponding authors)
                                                       KY4 and the standard-protocol legs
  battery   the above                                  Table 1's identical measurement battery
  real      data/ltown (public), data/city_d           L-Town benchmark; City D register chain
  figs      committed artifacts                        all main, Extended Data and SI figures + SI text
  llm       ollama runtime                             auditor stress test, dual-LLM run, audit
                                                       generalization (opt-in)
Steps whose inputs are absent are SKIPPED and reported, not failed.

Run ONE instance at a time. wntr writes EPANET scratch files (temp.inp/.bin/.rpt)
in this directory, so two concurrent twin-using processes overwrite each other and
simulations fail silently (EPANET Error 223); the driver itself is strictly serial.
"""
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ARGS = sys.argv[1:]
QUICK = "--quick" in ARGS
WITH_LLM = "--with-llm" in ARGS
ONLY = ARGS[ARGS.index("--only") + 1] if "--only" in ARGS and ARGS.index("--only") + 1 < len(ARGS) else None


def need(*paths):
    return all(os.path.exists(os.path.join(ROOT, p)) for p in paths)


SMOKE = [("tests/smoke_p1.py",), ("tests/smoke_p2p3.py",), ("tests/smoke_certificate.py",)]

# the BattLeDIM SCADA download lives outside the repo; public clones set this
LTOWN_SCADA = os.environ.get("LTOWN_SCADA_DIR", r"data/ltown/scada")

# (script, *args, required inputs...)  -- kept in dependency order
EXA7 = [
    ("eval/run_experiments.py",),        # results_full.json: headline, noise sweep, per-seed curves
    ("eval/run_full_stats.py",),         # results_stats.json: pooled 850-event stats, McNemar, confusion
    ("eval/baselines.py",),              # results_baselines.json: legacy trained comparators
    ("eval/baselines_fair.py",),         # results_baselines_fair.json: fair trained comparators (reads results_full)
    ("eval/conformal.py",),              # results_conformal.json: calibration + split-conformal comparator
    ("eval/contract_sensitivity.py",),   # results_sensitivity.json: 28-setting threshold sweep
    ("eval/ablation_components.py",),    # results_ablation_components.json
    ("eval/runtime_profile.py",),        # results_runtime.json (machine-dependent; hardware recorded)
]
TRANSFER = [
    ("eval/run_ky4.py", "data/ky4/ky4.inp"),
    ("eval/run_standard_net.py", "city_h", "data/city_h/city_h.inp"),
    ("eval/run_standard_net.py", "city_d", "data/city_d/city_d.inp"),
    ("eval/run_standard_net.py", "city_d_v2", "data/city_d/city_d.inp"),
    ("eval/run_city_d_std_controls.py", "data/city_d/city_d.inp"),
]
BATTERY = [("eval/run_unified_battery.py", "data/ky4/ky4.inp", "data/city_h/city_h.inp", "data/city_d/city_d.inp")]
REAL = [
    ("eval/run_ltown.py", "data/ltown/L-TOWN.inp", LTOWN_SCADA),
    ("eval/check_ltown_no_leak_window.py", "data/ltown/L-TOWN.inp", LTOWN_SCADA),
    ("eval/run_city_h.py", "data/city_h/city_h.inp"),
    ("eval/optimize_city_d_sensors.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_controls.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_adjacency_check.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_severity.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_ceiling_flow.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_flow_tier.py", "data/city_d/city_d.inp"),
    ("eval/run_city_d_demand_sensitivity.py", "data/city_d/city_d.inp"),
]
FIGS = [
    # Fig. 1 (figures/Fig1_infographic.*) and Fig. 2 (figures/Fig2_architecture.*)
    # are author-drawn schematics carrying no measured value; they are committed
    # assets, not generated here. make_fig1_overview.py still builds the earlier
    # generated overview, kept for reference.
    ("eval/make_fig1_overview.py",),           # superseded overview (not used by the paper)
    ("eval/make_partition_figs.py",),          # Fig. 3 + Fig. S3 (needs the network INPs)
    ("eval/make_fig3_exa7_evidence.py",),      # Fig. 4
    ("eval/make_fig4_realworld.py",),          # Fig. 5
    ("eval/make_fig5_method_pipeline.py",),    # Extended Data Fig. 1
    ("eval/make_fig6_method_contract.py",),    # Extended Data Fig. 2
    ("eval/make_figSI_networks.py",),          # Fig. S1
    ("eval/make_figures.py",),                 # Fig. S2 (multiseed robustness)
    ("eval/gen_si.py",),                       # Supplementary Information text + tables
]
LLM = [("eval/run_accountability.py",), ("eval/run_llm_executor.py",),
       ("eval/run_audit_generalization.py",)]

GROUPS = {"exa7": EXA7, "transfer": TRANSFER, "battery": BATTERY, "real": REAL,
          "figs": FIGS, "llm": LLM}


def split(step):
    """(script, *args, inputs) -> (cmd list, inputs list). Inputs are paths that exist
    under ROOT as directories or files; args are everything else."""
    script, rest = step[0], list(step[1:])
    inputs = [r for r in rest if r.startswith("data/") or os.path.isabs(r)]
    args = [r for r in rest if not (r.startswith("data/") or os.path.isabs(r))]
    return [PY, os.path.join(ROOT, script)] + args, inputs, script + (" " + " ".join(args) if args else "")


def run(step):
    cmd, inputs, label = split(step)
    print(f"\n>>> {label}")
    if inputs and not need(*inputs):
        print(f"    SKIP  (inputs not present: {inputs})")
        return "skip"
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    for f in ("temp.bin", "temp.inp", "temp.rpt"):     # EPANET scratch files
        try:
            os.remove(os.path.join(ROOT, f))
        except OSError:
            pass
    ok = r.returncode == 0
    print(f"    {'OK' if ok else 'FAIL'}  ({time.time()-t:.0f}s)")
    return "ok" if ok else "fail"


def main():
    print("=" * 64)
    print("REPRODUCE ALL: executor-supervisor accountable leak diagnosis")
    print(f"interpreter: {PY}")
    print("=" * 64)
    if not need("data/exa7/Exa7.inp", "data/exa7/partitions.pkl", "data/exa7/sensor_fingerprints.json"):
        print("MISSING vendored EXA7 inputs under data/exa7/")
        return 1
    if ONLY:
        if ONLY not in GROUPS:
            print("unknown group", ONLY, "; choose from", list(GROUPS))
            return 1
        steps = list(GROUPS[ONLY])
    elif QUICK:
        steps = SMOKE + FIGS
    else:
        steps = SMOKE + EXA7 + TRANSFER + BATTERY + REAL + FIGS + (LLM if WITH_LLM else [])
    results = [(split(s)[2], run(s)) for s in steps]
    print("\n" + "=" * 64 + "\nMANIFEST")
    for label, st in results:
        print(f"  [{st.upper():4s}] {label}")
    ok = all(st != "fail" for _, st in results)
    print("\n" + ("ALL STEPS OK" if ok else "SOME STEPS FAILED") + "\n" + "=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
