# -*- coding: utf-8 -*-
"""Does the language-model auditor catch anything a literal rule-checker cannot?

The paper's auditor prompt (agents/llm_client.py, LLMAuditor.SYSTEM) enumerates
five rejection rules, and each of the three corruption classes of the original
stress test (eval/run_accountability.py) trips exactly one of them. This
experiment asks the sharper question a referee will ask: on corruptions that
are NOT covered by any enumerated rule, does the model still reject, and does
it beat a ten-line rule-checker?

Design (same 16 clean ACCEPTED EXA7 packages as the original test, same seeds):
  auditors    RULES     the five prompt rules implemented literally in Python
              LLM/as-is the paper's prompt verbatim
              LLM/open  the paper's prompt plus one extra clause asking for any
                        other internal inconsistency (nothing enumerated)
  packages    16 clean, 3 x 16 in-taxonomy corruptions (A/B/C, controls),
              4 x 16 out-of-taxonomy corruptions, each of which passes every
              one of the five rules field-by-field but is jointly impossible:
              D  margin larger than the leak-existence probability
              E  an alternative listed with a higher posterior than the top
              F  posteriors summing to well over one
              G  an empty candidate region (nothing to dig)
  models      the paper's auditor deepseek-v4-pro:cloud (a family different from
              any executor component, as the paper requires) with the planner
              family gpt-oss:120b-cloud as a second opinion.
Every call is persisted to artifacts/llm_transcripts_audit_generalization.jsonl.

Run:  <wds_rag python> eval/run_audit_generalization.py
Output: artifacts/results_audit_generalization.json
"""
import os
import sys
import json
import copy
import time
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
from eval.run_accountability import corrupt as corrupt_in_taxonomy, NOISE, PERIOD, MAX_CLEAN

MODELS = ["deepseek-v4-pro:cloud", "gpt-oss:120b-cloud"]
PAPER_AUDITOR = "deepseek-v4-pro:cloud"   # the auditor of Table S9; gpt-oss:120b-cloud (the planner family) is the second opinion
N_STABILITY = 3          # temperature-0 repeats on a subset of out-of-taxonomy packages
IN_TAX = ["A_unsupported_confidence", "B_false_falsification", "C_low_evidence_dressed"]
OUT_TAX = ["D_margin_exceeds_existence", "E_alternative_outranks_top",
           "F_posteriors_exceed_one", "G_empty_region"]


# ----------------------------------------------------------------------------
# out-of-taxonomy corruptions: every field passes the five prompt rules on its own
# ----------------------------------------------------------------------------
def corrupt_out_of_taxonomy(summary, kind):
    s = copy.deepcopy(summary)
    if kind == "D_margin_exceeds_existence":
        # existence still >= 0.7 (rule 2), margin still >= 0.15 (rule 3), yet a
        # margin cannot exceed the leak-existence mass when the top is a leak
        s["leak_existence_probability"] = 0.72
        s["margin_over_next_best"] = 0.85
    elif kind == "E_alternative_outranks_top":
        # the "top" hypothesis is not the top: the first alternative carries more
        # posterior; margin left positive, so rule 3 still passes
        if s["alternatives"]:
            s["alternatives"][0]["posterior"] = round(min(0.99, s["top_hypothesis"]["posterior"] + 0.05), 4)
    elif kind == "F_posteriors_exceed_one":
        # posteriors must sum to at most one; give every alternative 0.35
        for a in s["alternatives"]:
            a["posterior"] = 0.35
    elif kind == "G_empty_region":
        # region <= R_max passes for zero nodes, but nothing can be dug
        s["top_hypothesis"]["candidate_region_nodes"] = 0
    return s


# ----------------------------------------------------------------------------
# the five prompt rules, literally
# ----------------------------------------------------------------------------
def rules_auditor(s):
    top = s["top_hypothesis"]
    if top["type"] != "leak":
        return {"reject": True, "reason": "rule1 top not leak"}
    if s["leak_existence_probability"] < 0.7:
        return {"reject": True, "reason": "rule2 existence<0.7"}
    if s["margin_over_next_best"] < 0.15:
        return {"reject": True, "reason": "rule3 margin<0.15"}
    if top["fit_residual_mahalanobis"] > 3:
        return {"reject": True, "reason": "rule4 residual>3"}
    for a in s["alternatives"]:
        if a["status"] == "falsified" and a["residual_mahalanobis"] < 1.5:
            return {"reject": True, "reason": "rule5 falsified alt fits"}
    return {"reject": False, "reason": "no rule fired"}


def rules_pass_all(s):
    return not rules_auditor(s)["reject"]


class OpenAuditor(LLMAuditor):
    SYSTEM = LLMAuditor.SYSTEM.replace(
        "Otherwise reject=false.",
        " 6. the summary is internally inconsistent in ANY other way: quantities that "
        "cannot jointly hold in a valid probabilistic evidence summary, or a "
        "recommendation that could not be acted on.\nOtherwise reject=false.")


def collect_clean(ctx):
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
    clean = []
    for pid in ctx.partitions[:12]:
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
            if evaluate_contract(ev, contract).decision == "ACCEPT":
                clean.append(summarize_evidence(ev))
            if len(clean) >= MAX_CLEAN:
                break
        if len(clean) >= MAX_CLEAN:
            break
    return clean


def main():
    print("=" * 76)
    print("AUDIT GENERALIZATION: rule-checker vs language-model auditor")
    print("=" * 76)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    clean = collect_clean(ctx)
    print(f"clean ACCEPTED packages: {len(clean)}")

    # package sets ---------------------------------------------------------
    sets = {"clean": [(i, s) for i, s in enumerate(clean)]}
    for k in IN_TAX:
        sets[k] = [(i, corrupt_in_taxonomy(s, k)) for i, s in enumerate(clean)]
    for k in OUT_TAX:
        sets[k] = [(i, corrupt_out_of_taxonomy(s, k)) for i, s in enumerate(clean)]

    # sanity: out-of-taxonomy packages must pass every literal rule
    for k in OUT_TAX:
        n_pass = sum(rules_pass_all(s) for _, s in sets[k])
        print(f"   {k:30s}: {n_pass}/{len(sets[k])} pass all five rules field-by-field")
        assert n_pass == len(sets[k]), "out-of-taxonomy corruption tripped a rule; redesign it"

    results = {"provenance": {
        "network": "exa7", "noise_m": NOISE, "n_clean": len(clean),
        "clean_selection": "identical to eval/run_accountability.py (seeds 42/43, first 12 partitions, 20 and 50 L/s)",
        "paper_auditor_model": PAPER_AUDITOR, "models_used": MODELS,
        "in_taxonomy_classes": IN_TAX, "out_of_taxonomy_classes": OUT_TAX,
        "open_prompt_extra_clause": OpenAuditor.SYSTEM.split(" 6. ")[1].split("\nOtherwise")[0],
        "n_stability_repeats": N_STABILITY,
    }, "rules": {}, "llm": {}}

    # RULES auditor --------------------------------------------------------
    for k, items in sets.items():
        rej = [rules_auditor(s)["reject"] for _, s in items]
        results["rules"][k] = {"n": len(rej), "rejected": int(sum(rej))}
    print("\nRULES auditor:", {k: f"{v['rejected']}/{v['n']}" for k, v in results["rules"].items()})

    # LLM auditors ---------------------------------------------------------
    tpath = os.path.join(ROOT, "artifacts", "llm_transcripts_audit_generalization.jsonl")
    for model in MODELS:
        client = LLMClient(model, temperature=0.0, num_predict=1024, transcript_path=tpath, timeout=300)
        if not client.ok:
            print("LLM client unavailable; aborting"); return 1
        for label, aud in (("as_is", LLMAuditor(client)), ("open", OpenAuditor(client))):
            key = f"{model}|{label}"
            print(f"\n[{key}]")
            block = {}
            for k, items in sets.items():
                decisions, reasons = [], []
                t0 = time.time()
                for _, s in items:
                    r = aud.audit_summary(s)
                    decisions.append(bool(r["reject"]))
                    reasons.append(r["reason"])
                block[k] = {"n": len(decisions), "rejected": int(sum(decisions)),
                            "decisions": decisions, "reasons": reasons}
                print(f"   {k:30s}: rejected {sum(decisions)}/{len(decisions)}  ({time.time()-t0:.0f}s)")
            # stability on 8 out-of-taxonomy packages (2 per class), N repeats
            stab = []
            for k in OUT_TAX:
                for _, s in sets[k][:2]:
                    decs = [int(aud.audit_summary(s)["reject"]) for _ in range(N_STABILITY)]
                    stab.append(max(decs.count(0), decs.count(1)) / len(decs))
            block["_stability_out_of_taxonomy"] = float(np.mean(stab)) if stab else None
            print(f"   stability (out-of-taxonomy, {N_STABILITY} repeats): {block['_stability_out_of_taxonomy']:.3f}")
            results["llm"][key] = block

    outp = os.path.join(ROOT, "artifacts", "results_audit_generalization.json")
    json.dump(results, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {outp}")

    # headline table
    print("\n%-34s %-8s %-8s %-8s %-8s" % ("catch / false-alarm", "clean", "in-tax", "out-tax", ""))
    def agg(block, keys):
        n = sum(block[k]["n"] for k in keys); r = sum(block[k]["rejected"] for k in keys)
        return f"{r}/{n}"
    print("%-34s %-8s %-8s %-8s" % ("RULES (five prompt rules)", agg(results["rules"], ["clean"]),
                                    agg(results["rules"], IN_TAX), agg(results["rules"], OUT_TAX)))
    for key, block in results["llm"].items():
        print("%-34s %-8s %-8s %-8s" % (key, agg(block, ["clean"]), agg(block, IN_TAX), agg(block, OUT_TAX)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
