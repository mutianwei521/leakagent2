"""
Dual-LLM, different-families demonstration (item 2): an LLM EXECUTOR planner
(gpt-oss:120b) proposes the diagnostic plan and an independent LLM SUPERVISOR
(deepseek-v4-pro) audits — two different model families, with all numbers from the
twin. We compare the LLM-orchestrated pipeline's decisions against the fully
deterministic pipeline on a small mixed sample (real LLM calls).

Run:  python eval/run_llm_executor.py     (via the wds_rag interpreter; needs ollama)
"""
import os
import sys
import json
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
from agents.llm_client import LLMClient, LLMAuditor, LLMExecutorPlanner
from eval.run_full_stats import gen_scenarios

EXEC_MODEL, SUP_MODEL = "gpt-oss:120b-cloud", "deepseek-v4-pro:cloud"
NOISE, N_CASES = 0.05, 16


def decide(ev, contract, sup=None):
    v = evaluate_contract(ev, contract)
    if v.decision == "ACCEPT" and sup is not None:
        a = sup.audit(ev, contract)
        if a.get("reject"):
            v.decision = "REJECT"
    acted = v.decision == "ACCEPT"
    pred = ev.top1.partition if (acted and ev.top1.kind == "leak") else None
    return acted, pred


def main():
    print("=" * 70)
    print(f"DUAL-LLM PIPELINE  executor={EXEC_MODEL}  supervisor={SUP_MODEL}")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE)
    contract = GoalContract(min_leak_probability=0.7, min_top1_margin=0.15,
                            max_candidate_region_nodes=60, alt_falsify_prob=0.20)

    tp = os.path.join(ROOT, "artifacts", "llm_transcripts_executor.jsonl")
    planner_client = LLMClient(EXEC_MODEL, temperature=0.0, num_predict=384, transcript_path=tp)
    sup_client = LLMClient(SUP_MODEL, temperature=0.0, num_predict=256, transcript_path=tp)
    if not (planner_client.ok and sup_client.ok):
        print("LLM unavailable; aborting.")
        return 1
    planner = LLMExecutorPlanner(planner_client)
    auditor = LLMAuditor(sup_client)

    rng = np.random.default_rng(2024)
    scen = gen_scenarios(ctx, gr, np.random.default_rng(42))
    rng.shuffle(scen)
    scen = scen[:N_CASES]

    agree, correct_llm, correct_det, plans = 0, 0, 0, []
    for i, s in enumerate(scen):
        true_leak = s["true_class"] == "leak"
        arng = np.random.default_rng(1000 + i)
        # deterministic pipeline
        evd = EvidencePackage(observation=s["observation"], demand_period="day_normal", network=ctx.name)
        ex.set_plan(None); ex.step(evd, 1); ex.active_sense_verify(evd, s.get("full_dp", {}), "day_normal", arng, asens)
        act_d, pred_d = decide(evd, contract, sup=None)
        # LLM-orchestrated pipeline: gpt-oss plans, deepseek audits
        rtop = [int(p) for p, _ in gr.rank_partitions(s["observation"], top_k=5)]
        plan = planner.plan(s["observation"], ctx.sensors, rtop)
        plans.append(plan)
        evl = EvidencePackage(observation=s["observation"], demand_period="day_normal", network=ctx.name)
        ex.set_plan(plan); ex.step(evl, 1); ex.active_sense_verify(evl, s.get("full_dp", {}), "day_normal",
                                                                   np.random.default_rng(1000 + i), asens)
        act_l, pred_l = decide(evl, contract, sup=auditor)
        ex.set_plan(None)

        ok_d = (act_d and true_leak and pred_d == s["true_region"]) or (not act_d and not true_leak)
        ok_l = (act_l and true_leak and pred_l == s["true_region"]) or (not act_l and not true_leak)
        correct_det += int(ok_d); correct_llm += int(ok_l); agree += int(act_d == act_l)
        print(f"  [{i+1:2d}] true={s['true_class'][:6]:6s} det={'ACT' if act_d else 'ABS'} "
              f"llm={'ACT' if act_l else 'ABS'}  plan(dem={plan['include_demand']},"
              f"sen={plan['include_sensor']},val={plan['include_valve']})")

    n = len(scen)
    out = {"executor_model": EXEC_MODEL, "supervisor_model": SUP_MODEL, "n_cases": n,
           "note": "two different LLM families; numbers from twin; small demo",
           "decision_agreement_llm_vs_deterministic": agree / n,
           "correct_decisions_deterministic": correct_det / n,
           "correct_decisions_dual_llm": correct_llm / n,
           "sample_plans": plans[:6]}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_llm_executor.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"\n--- DUAL-LLM (different families) on {n} mixed cases ---")
    print(f"  decision agreement (LLM vs deterministic): {agree/n*100:.0f}%")
    print(f"  correct decisions: deterministic {correct_det/n*100:.0f}%  | dual-LLM {correct_llm/n*100:.0f}%")
    print("saved -> artifacts/results_llm_executor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
