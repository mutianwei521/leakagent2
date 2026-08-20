"""
LLM layer: an independent SUPERVISOR auditor (and an optional independent EXECUTOR
planner of a different model family).

Design for non-collusion / non-sycophancy (roadmap 3.5):
  - The auditor sees ONLY a structured numeric evidence summary, never the
    (here deterministic) executor's internal reasoning or persuasive prose.
  - It is a DIFFERENT model family from anything used by the executor.
  - It can only ADD a rejection on top of the deterministic goal contract; it can
    never flip a deterministically-failed hard check to ACCEPT.
  - Its job is an internal-consistency / sufficiency check that is verifiable from
    the numbers (e.g. "claimed a confident leak but its residual does not fit").

Backend: the local **ollama** runtime is called DIRECTLY (no langchain shim, which
proved version-fragile). Every model reply is requested as strict JSON via ollama's
`format="json"` constraint, and every call (model, messages, raw reply, parsed
result) is appended to a transcript file so the accountability metrics can be
regenerated from cached transcripts without live model access.
"""
from __future__ import annotations
import json
import os
import re

try:
    import ollama
    _HAVE_LLM = True
except Exception:                       # pragma: no cover
    _HAVE_LLM = False

_ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def _parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    for attempt in (blob, re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", blob))):
        try:
            return json.loads(attempt)
        except Exception:
            continue
    return None


class LLMClient:
    """Direct ollama chat client. `format='json'` forces the model to emit valid
    JSON (essential for reasoning models such as gpt-oss, whose prose otherwise does
    not parse). `num_predict` must be large enough that the JSON is not truncated."""

    def __init__(self, model, temperature=0.0, num_predict=384,
                 base_url="http://localhost:11434", transcript_path=None, timeout=180):
        self.model = model
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.ok = _HAVE_LLM
        self._client = ollama.Client(host=base_url, timeout=timeout) if _HAVE_LLM else None
        self.transcript_path = transcript_path or os.path.join(_ARTIFACTS, "llm_transcripts.jsonl")

    def _log(self, system, user, raw, parsed, error=None):
        try:
            os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": self.model, "temperature": self.temperature,
                                    "system": system, "user": user, "raw": raw,
                                    "parsed": parsed, "error": error}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def json_invoke(self, system, user, retries=1):
        if not self.ok:
            return None
        for _ in range(retries + 1):
            try:
                resp = self._client.chat(
                    model=self.model, format="json",
                    options={"temperature": self.temperature, "num_predict": self.num_predict},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                content = resp["message"]["content"]
                d = _parse_json(content)
                self._log(system, user, content, d)
                if d is not None:
                    return d
            except Exception as e:
                self._log(system, user, None, None, error=repr(e))
        return None


def summarize_evidence(ev) -> dict:
    """The ONLY thing the LLM auditor sees — pure numbers, no reasoning."""
    top1 = ev.top1
    rr = (top1.params.get("residual") if top1 else {}) or {}
    alts = []
    for h in sorted(ev.hypotheses, key=lambda h: -h.posterior):
        if h is top1:
            continue
        ar = (h.params.get("residual") or {})
        alts.append({"type": h.kind,
                     "partition": h.partition if h.kind in ("leak", "demand_anomaly") else None,
                     "posterior": round(h.posterior, 4),
                     "residual_mahalanobis": round(ar.get("mahalanobis_per_dof", 999.0), 3),
                     "status": h.status})
        if len(alts) >= 5:
            break
    return {
        "top_hypothesis": {
            "type": top1.kind if top1 else "none",
            "partition": getattr(top1, "partition", None),
            "posterior": round(top1.posterior, 4) if top1 else 0.0,
            "fit_residual_mahalanobis": round(rr.get("mahalanobis_per_dof", 999.0), 3),
            "candidate_region_nodes": len(ev.candidate_region),
        },
        "leak_existence_probability": round(ev.existence_mass, 4),
        "margin_over_next_best": round(ev.margin, 4),
        "alternatives": alts,
        "demand_period": ev.demand_period,
    }


class LLMAuditor:
    SYSTEM = (
        "You are an INDEPENDENT safety supervisor for a water-distribution-network "
        "leak-diagnosis system. You receive ONLY a structured numeric evidence summary "
        "(never the diagnostic agent's reasoning). Decide whether the evidence SUFFICIENTLY "
        "and CONSISTENTLY supports dispatching a repair crew to the claimed leak partition.\n"
        "Residual Mahalanobis-per-dof near 1.0 means a hypothesis REPRODUCES the observation; "
        ">>3 means it does NOT fit. Reject (reject=true) if ANY hold:\n"
        " 1. top_hypothesis.type is not 'leak';\n"
        " 2. leak_existence_probability < 0.7;\n"
        " 3. margin_over_next_best < 0.15;\n"
        " 4. top_hypothesis.fit_residual_mahalanobis > 3 (claimed leak does not fit);\n"
        " 5. any alternative marked status='falsified' but with residual_mahalanobis < 1.5 "
        "(it actually fits well -> inconsistent, the diagnosis is not trustworthy).\n"
        "Otherwise reject=false. Output STRICT JSON: {\"reject\": true|false, \"reason\": \"<short>\"}."
    )

    def __init__(self, client: LLMClient):
        self.client = client

    def audit_summary(self, summary: dict):
        user = "Evidence summary:\n" + json.dumps(summary, ensure_ascii=False, indent=2) \
               + "\n\nReturn STRICT JSON {\"reject\": true|false, \"reason\": \"...\"}."
        d = self.client.json_invoke(self.SYSTEM, user)
        if d is None or "reject" not in d:
            return {"reject": True, "reason": "auditor_parse_failure_default_safe"}
        return {"reject": bool(d["reject"]), "reason": str(d.get("reason", ""))[:200]}

    def audit(self, ev, contract):    # SupervisorAgent integration hook
        return self.audit_summary(summarize_evidence(ev))


class LLMExecutorPlanner:
    """Optional LLM-orchestrated executor (a DIFFERENT model family from the
    supervisor): given a numeric observation summary it proposes the diagnostic
    PLAN, which non-leak hypothesis families to test and a short rationale, while
    all numbers still come from the twin (anti-fabrication). Leak zones are always
    tested exhaustively, so localization is never harmed by the LLM."""
    SYSTEM = (
        "You are the orchestration head of a water-network leak-diagnosis agent. "
        "Given a numeric summary of a pressure-deviation observation, decide which "
        "NON-LEAK explanations are worth testing alongside the leak hypotheses: a "
        "DEMAND anomaly (a few-to-many sensors depressed broadly across a zone), a "
        "SENSOR fault (one or two channels anomalous while the rest read ~0), or a "
        "VALVE mis-state (large, structured, network-wide shifts). Always allow the "
        "leak hypotheses. Keep the rationale to at most 12 words. Output STRICT JSON: "
        "{\"include_demand\": bool, \"include_sensor\": bool, \"include_valve\": bool, "
        "\"rationale\": \"<<=12 words>\"}."
    )

    def __init__(self, client: LLMClient):
        self.client = client

    def plan(self, observation, sensors, retrieval_top):
        vals = [(s, round(float(observation.get(s, 0.0)), 3)) for s in sensors]
        vals.sort(key=lambda t: abs(t[1]), reverse=True)
        n_anom = sum(1 for _, v in vals if abs(v) > 0.2)
        summary = {
            "n_sensors": len(sensors),
            "n_anomalous_sensors(|dp|>0.2m)": n_anom,
            "top_sensor_deviations_m": vals[:6],
            "max_abs_dp_m": round(max((abs(v) for _, v in vals), default=0.0), 3),
            "retrieval_top_zones": retrieval_top,
        }
        d = self.client.json_invoke(self.SYSTEM, "Observation summary:\n"
                                    + json.dumps(summary, ensure_ascii=False, indent=2)
                                    + "\n\nReturn STRICT JSON.")
        if not d:
            return {"include_demand": True, "include_sensor": True, "include_valve": True,
                    "rationale": "planner_parse_failure_default_all"}
        return {"include_demand": bool(d.get("include_demand", True)),
                "include_sensor": bool(d.get("include_sensor", True)),
                "include_valve": bool(d.get("include_valve", True)),
                "rationale": str(d.get("rationale", ""))[:200]}
