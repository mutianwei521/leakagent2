"""
Smoke test for the acceptance certificate: an ACCEPTED diagnosis issues a
certificate that verifies against its evidence, and any tampering with the
evidence (or the certificate's recorded decision) breaks verification.

Run:  python tests/smoke_certificate.py   (via the wds_rag interpreter)
"""
import os
import sys
import copy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.contract import GoalContract
from schemas.certificate import evidence_digest
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent
from run_diagnosis import diagnose
from anomaly_sim.leak_anomaly import make_leak_scenario

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def main():
    print("=" * 66)
    print("CERTIFICATE SMOKE TEST")
    print("=" * 66)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    noise = 0.02
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=noise, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=50, noise_std=noise)
    sv = SupervisorAgent(GoalContract(min_leak_probability=0.55, min_top1_margin=0.12,
                                      max_candidate_region_nodes=60, alt_falsify_prob=0.20))
    rng = np.random.default_rng(1)

    pid = ctx.partitions[0]
    node = gr.candidate_nodes(pid, max_nodes=1)[0]
    s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, node, 20.0,
                           "day_normal", noise_std=noise, rng=rng)
    res = diagnose(s["observation"], ctx, ex, sv, "day_normal", case_id="cert-demo")
    print(f"outcome={res.outcome} accepted_partition={res.accepted_partition}")

    check("ACCEPTED leak issues a certificate", res.acted and res.certificate is not None)
    cert = res.certificate
    if cert is not None:
        print(f"  evidence_sha256={cert.evidence_sha256[:16]}...  "
              f"accepted={cert.accepted_hypothesis['type']}:P{cert.accepted_hypothesis['partition']}")
        check("certificate verifies against its evidence", cert.verify(res.evidence))

        # tamper 1: alter the evidence after issuance -> verification must fail
        tampered = copy.deepcopy(res.evidence)
        if tampered.top1 is not None:
            tampered.top1.partition = (tampered.top1.partition + 1) % len(ctx.partitions)
        check("tampering with the evidence breaks verification",
              not cert.verify(tampered),
              f"(orig {cert.evidence_sha256[:8]} vs tampered {evidence_digest(tampered)[:8]})")

        # reproducibility: re-evaluating the contract on the evidence reproduces the certificate
        from agents.goal_contract import evaluate_contract
        v2 = evaluate_contract(res.evidence, sv.contract)
        check("contract re-evaluation on the evidence reproduces the certificate",
              v2.decision == "ACCEPT" and v2.checks == cert.predicate_checks)

    print("\n" + "=" * 66)
    ok = all(results)
    print(f"RESULT: {'ALL PASS' if ok else 'SOME FAILED'}  ({sum(results)}/{len(results)})")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
