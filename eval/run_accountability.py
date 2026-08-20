"""
C4 accountability experiment (real LLM calls, EXA7):
  - The deterministic executor produces ACCEPTED leak packages (clean).
  - We synthesize CORRUPTED packages that are internally INCONSISTENT (a confident
    claim whose residual does not fit / a 'falsified' alternative that actually fits).
  - An INDEPENDENT LLM supervisor (deepseek-v4-pro, a different family from any executor
    model) audits the structured summary ONLY and must catch the corruptions.
  - Decision stability: N repeats at temperature 0 -> agreement rate (reproducibility).

Run:  python eval/run_accountability.py    (via the wds_rag interpreter; needs ollama)
"""
import os
import sys
import json
import copy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.evidence import EvidencePackage
from schemas.contract import GoalContract
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from tools.active_sensing import ActiveSensingTool
from agents.executor import ExecutorAgent
from agents.goal_contract import evaluate_contract
from agents.llm_client import LLMClient, LLMAuditor, summarize_evidence
from anomaly_sim.leak_anomaly import make_leak_scenario

SUPERVISOR_MODEL = "deepseek-v4-pro:cloud"
NOISE = 0.05
PERIOD = "day_normal"
MAX_CLEAN = 16
N_STABILITY = 8


def corrupt(summary, kind):
    s = copy.deepcopy(summary)
    if kind == "A_unsupported_confidence":      # confident leak that does not fit
        s["top_hypothesis"]["fit_residual_mahalanobis"] = 8.4
    elif kind == "B_false_falsification":       # an alt that fits is marked falsified
        if s["alternatives"]:
            s["alternatives"][0]["status"] = "falsified"
            s["alternatives"][0]["residual_mahalanobis"] = 0.7
        s["top_hypothesis"]["fit_residual_mahalanobis"] = 4.2
    elif kind == "C_low_evidence_dressed":      # high posterior but low existence + tiny margin
        s["leak_existence_probability"] = 0.42
        s["margin_over_next_best"] = 0.04
    return s


def main():
    print("=" * 72)
    print("C4 ACCOUNTABILITY — independent LLM supervisor audit (EXA7)")
    print(f"supervisor model = {SUPERVISOR_MODEL}")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt = DigitalTwinTool()
    asens = ActiveSensingTool()
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=50, noise_std=NOISE)
    contract = GoalContract(min_leak_probability=0.7, min_top1_margin=0.15,
                            max_candidate_region_nodes=60, alt_falsify_prob=0.20)
    rng = np.random.default_rng(42)
    arng = np.random.default_rng(43)

    # ---- collect CLEAN accepted leak packages ----------------------------
    clean = []
    parts = ctx.partitions[:12]
    for pid in parts:
        node = gr.candidate_nodes(pid, max_nodes=1)
        node = node[0] if node else None
        if node is None:
            continue
        for rate in (20.0, 50.0):
            s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                                   node, rate, PERIOD, noise_std=NOISE, rng=rng)
            if not s:
                continue
            ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
            ex.step(ev, 1)
            ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens)
            v = evaluate_contract(ev, contract)
            if v.decision == "ACCEPT":
                clean.append(summarize_evidence(ev))
            if len(clean) >= MAX_CLEAN:
                break
        if len(clean) >= MAX_CLEAN:
            break
    print(f"collected {len(clean)} clean ACCEPTED packages")

    client = LLMClient(SUPERVISOR_MODEL, temperature=0.0, num_predict=256,
                       transcript_path=os.path.join(ROOT, "artifacts", "llm_transcripts_accountability.jsonl"))
    auditor = LLMAuditor(client)
    if not client.ok:
        print("LLM unavailable; aborting C4 experiment.")
        return 1

    # ---- false-alarm on clean -------------------------------------------
    print("\n[A] auditing CLEAN packages (expect reject=false)...")
    clean_rej = 0
    for i, s in enumerate(clean):
        r = auditor.audit_summary(s)
        clean_rej += int(r["reject"])
        print(f"   clean {i+1}/{len(clean)}: reject={r['reject']}")
    false_alarm = clean_rej / len(clean) if clean else 0.0

    # ---- catch-rate on corrupted ----------------------------------------
    print("\n[B] auditing CORRUPTED packages (expect reject=true)...")
    kinds = ["A_unsupported_confidence", "B_false_falsification", "C_low_evidence_dressed"]
    caught, total = {k: 0 for k in kinds}, {k: 0 for k in kinds}
    for i, s in enumerate(clean):
        kind = kinds[i % len(kinds)]
        cs = corrupt(s, kind)
        r = auditor.audit_summary(cs)
        total[kind] += 1
        caught[kind] += int(r["reject"])
        print(f"   corrupt {i+1} [{kind}]: reject={r['reject']}")
    catch_overall = sum(caught.values()) / max(sum(total.values()), 1)

    # ---- decision stability (temp 0, N repeats) -------------------------
    print(f"\n[C] decision stability ({N_STABILITY} repeats @ temp 0) ...")
    stab = []
    for s in clean[:5]:
        decs = [int(auditor.audit_summary(s)["reject"]) for _ in range(N_STABILITY)]
        agree = max(decs.count(0), decs.count(1)) / len(decs)
        stab.append(agree)
    stability = float(np.mean(stab)) if stab else 0.0

    out = {
        "supervisor_model": SUPERVISOR_MODEL, "n_clean": len(clean),
        "false_alarm_rate_on_clean": false_alarm,
        "catch_rate_overall_on_corrupted": catch_overall,
        "catch_rate_by_type": {k: (caught[k] / total[k] if total[k] else None) for k in kinds},
        "decision_stability_temp0": stability,
        "note": "real LLM audits; auditor sees only the numeric summary (anti-collusion).",
    }
    outp = os.path.join(ROOT, "artifacts", "results_accountability.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=2)

    print("\n--- C4 RESULT ---")
    print(f"  false-alarm on clean      : {false_alarm*100:.1f}%  (lower better)")
    print(f"  catch-rate on corrupted   : {catch_overall*100:.1f}%  (higher better)")
    for k in kinds:
        print(f"     {k:28s}: {out['catch_rate_by_type'][k]}")
    print(f"  decision stability (temp0): {stability*100:.1f}%")
    print(f"saved -> {outp}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
