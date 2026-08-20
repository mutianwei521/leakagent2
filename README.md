<h1 align="center">LeakAgent-2</h1>

<p align="center"><b>Verifiable abstention makes AI leak diagnosis accountable in water distribution networks</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.18836"><img src="https://img.shields.io/badge/arXiv-2608.18836-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/hydraulics-EPANET%20%2F%20WNTR-0a7bbb.svg" alt="EPANET/WNTR">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

<p align="center">
  <img src="figures/Fig2_architecture.png" width="85%" alt="Executor-supervisor architecture">
</p>

Water utilities rarely trust AI leak localizers to dispatch repair crews, and they are right
not to: a model that answers on every event cannot justify an excavation. **LeakAgent-2
recasts leak localization as decision-making under verifiable abstention.** A
physics-grounded **executor** agent falsifies competing hypotheses (leak, demand anomaly,
sensor fault, valve mis-state) against an EPANET/WNTR digital twin; an independent
**supervisor** agent checks the numeric evidence against a code-verifiable **goal contract**
and then does exactly one of three things: **certifies a dispatch** (with a tamper-evident
SHA-256 certificate), **requests more evidence** (discriminative active sensing), or
**abstains** (with a dossier explaining exactly which predicate failed). An optional LLM
auditor from a different model family can tighten the gate but can never overturn a failed
hard check.

## Highlights

- **Accountability, not just accuracy.** Every ACT decision carries a certificate binding
  the decision, thresholds, per-predicate values and evidence digest; every ABSTAIN names
  the failed predicate. Abstention is a first-class outcome, never a failure mode.
- **A falsification loop, not a point estimate.** Hypotheses must reproduce the observed
  pressures in the digital twin; residual Mahalanobis gates and an Occam penalty eliminate
  over-flexible explanations before any decision is made.
- **The same measurement battery on five networks.** One benchmark (EXA7), two public
  transfer networks (KY4, BattLeDIM L-Town) and two utility networks (City H, City D),
  including a 194-event audited field register.
- **A goal contract you can read.** Acceptance is a conjunction of arithmetic predicates
  (existence, region size, margin, alternative elimination, residual consistency, safety),
  swept to give the full risk-coverage frontier.
- **Integrity by construction.** No reported metric is a hard-coded literal
  (`tests/integrity_lint.py` enforces this), every figure value is read from a committed
  results file, and all LLM calls are persisted as replayable transcripts.

## Results at a glance

The identical battery, applied to all five networks (Table 1 of the paper; fractions in
`artifacts/`):

| | EXA7 | KY4 | City H | City D | L-Town | City D register |
|---|---|---|---|---|---|---|
| Data tier | in-silico benchmark | in-silico transfer | in-silico transfer | in-silico transfer | third-party benchmark | audited field register |
| Forced top-1 | 81.7% | 88.7% | 66.7% | 44.7% | 15% | 12% |
| Coverage at operating point | 40.5% | 34.4% | 25.6% | 20.0% | 12% | 2.6% excavation; 44% survey |
| Decision precision on acted | 96.0% | 96.3% | 91.5% | 81.8% | 100% (4/4) | 60% excavation; 100% district survey |
| No-leak controls (false dispatch) | 0/50 | 0/50 | 0/50 | 2/50 | n/a | 1/50 |

The pattern is the point: as networks get harder, the system **abstains more instead of
guessing more**, and the precision of what it does act on stays defensible. On the field
register, where every leak signature sits below the sensor noise floor, the honest answer
is near-total abstention plus a flow-balance survey tier that recovers 44% of events at
100% district precision.

> All in-silico results are pooled over 5 seeds at sigma = 0.05 m sensor noise; the L-Town
> and register columns are exact counts. See the paper for confidence intervals and the
> full protocol.

## Installation

```bash
git clone https://github.com/mutianwei521/leakagent2.git
cd leakagent2
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements-min.txt
```

`requirements-min.txt` installs the core stack (WNTR/EPANET, NumPy/SciPy, leidenalg,
matplotlib). The exact frozen environment used for the paper is in `requirements.txt`;
the trained neural baselines additionally need PyTorch and torch-geometric.

Verify the physics core in about a minute:

```bash
python tests/smoke_p1.py          # twin falsification: right node fits, wrong zone does not
python tests/smoke_p2p3.py        # full executor-supervisor loop incl. forced abstentions
python tests/smoke_certificate.py # tamper-evident acceptance certificate
python tests/integrity_lint.py    # no metric anywhere is a hard-coded literal
```

## Quick start: diagnose one event

```python
import numpy as np
from data.context import build_context
from schemas.contract import GoalContract
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent
from run_diagnosis import diagnose
from anomaly_sim.leak_anomaly import make_leak_scenario

ctx = build_context("configs/exa7.yaml")               # vendored benchmark, self-contained
contract = GoalContract()                              # default acceptance thresholds (Eq. 4)
noise = 0.02
gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
executor = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                         ResidualTool(sensor_std=noise,
                                      pass_threshold=contract.max_residual_mahalanobis),
                         BayesianEngine(), top_k_leak=3, noise_std=noise)
supervisor = SupervisorAgent(contract)

# a 20 L/s leak at a known node, observed through 30 noisy sensors
leak_node = gr.candidate_nodes(ctx.partitions[0], max_nodes=1)[0]
s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors,
                       leak_node, 20.0, "day_normal", noise_std=noise,
                       rng=np.random.default_rng(7))
result = diagnose(s["observation"], ctx, executor, supervisor, "day_normal")

print(result.outcome)              # 'ACTED' or 'ABSTAINED'
print(result.accepted_partition)   # zone to dispatch to, if ACTED
print(result.certificate)          # SHA-256 acceptance certificate, if ACTED
```

`tests/smoke_p2p3.py` runs this loop across a leak, a sensor fault, a demand anomaly and a
micro leak, asserting the accountable behaviour of each.

## Reproducing the paper

One command re-runs every physics and statistics number and every generated figure, in
dependency order:

```bash
python reproduce_all.py            # full run (EXA7 leg + transfer + battery + real + figures)
python reproduce_all.py --quick    # smoke tests + figures + SI from the committed artifacts
python reproduce_all.py --only figs
python reproduce_all.py --with-llm # adds the three LLM experiments (needs an ollama runtime)
```

| Group | What it regenerates | Needs |
|---|---|---|
| `exa7` | headline metrics, noise sweep, baselines, conformal comparator, ablations, sensitivity | vendored `data/exa7/` (included) |
| `transfer` | KY4 / City H / City D standard-protocol runs | `data/ky4/` (included); City H/City D models on request |
| `battery` | the cross-network battery behind Table 1 | same as above |
| `real` | L-Town leg and the City D register leg | `data/ltown/` (included); City D model on request |
| `figs` | every generated figure + Supplementary Information | committed `artifacts/` only |
| `llm` | LLM auditor and dual-LLM experiments (opt-in) | ollama with the pinned model tags |

Three things to know:

1. **Steps whose inputs are absent are skipped and reported, never faked.** With the public
   data alone, the EXA7, KY4 and L-Town legs and all figures reproduce end to end.
2. **Runs are strictly serial by design.** Concurrent WNTR/EPANET runs in one working
   directory corrupt each other's scratch files; `reproduce_all.py` enforces the order.
3. **Every quoted number has a provenance path.** `eval/paper_numbers.py` prints each value
   the paper quotes, as the paper rounds it, straight from `artifacts/`, so a regeneration
   can be diffed line by line. `eval/make_source_data.py` builds the journal Source Data
   workbooks from the same files.

The two rebuildable simulation caches (`data/ky4/leak_cache.pkl`,
`data/ltown/leak_cache.pkl`, about 85 MB together) are not committed; rebuild them once with

```bash
python data/ky4_build_library.py && python data/ltown_build_library.py
```

For the L-Town leg, download the BattLeDIM 2020 SCADA dataset from the competition site
(see `data/ltown/ATTRIBUTION.md`) and either place it under `data/ltown/scada/` or point
the scripts at it:

```bash
export LTOWN_SCADA_DIR=/path/to/battledim/ltown
```

## The LLM experiments (opt-in, replayable)

The deterministic pipeline never needs a language model. Three experiments quantify what an
LLM layer adds and are run only with `--with-llm`:

- **Independent auditor stress test**: a supervisor-side auditor (different model family,
  temperature 0, strict JSON, sees only the numeric evidence summary) must catch corrupted
  evidence packages. Fails safe: any parse failure is a rejection.
- **Dual-LLM run**: an executor-side planner proposes which confounder families to test;
  leak zones are always tested exhaustively, so the planner can never harm localization.
- **Out-of-taxonomy audit generalization**: four corruption classes the auditor was never
  prompted about.

Every call is committed to `artifacts/llm_transcripts_*.jsonl`, so all audit metrics can be
recomputed from the transcripts without any model access.

## Repository layout

```
run_diagnosis.py        the executor <-> supervisor loop (library entry point)
reproduce_all.py        one-command reproduction driver
agents/                 executor, supervisor, goal contract, LLM client/auditor/planner
schemas/                evidence package, goal contract thresholds, acceptance certificate
tools/                  digital twin, residual gate, Bayesian fusion, retrieval, active sensing
anomaly_sim/            leak + confounder scenario generators (demand, sensor, valve)
data/                   per-network contexts, setup and cache builders; vendored EXA7/KY4/L-Town
configs/                one YAML per network leg
eval/                   experiment runners, metrics, figure and SI generators
tests/                  smoke tests + the integrity lint
artifacts/              committed results JSON + LLM transcripts (the provenance snapshot)
vendor/                 the prior-generation retrieval pipeline, kept as the baseline
docs/                   plain-language tutorial with worked examples
```

## Data availability

| Network | Model | Scenarios / labels | In this repo |
|---|---|---|---|
| EXA7 | vendored | generated | yes, fully self-contained |
| KY4 | public (WNTR library, Kentucky dataset) | generated | yes |
| L-Town | public (BattLeDIM 2020, KIOS) | competition SCADA + labels | model + derived inputs; SCADA from the competition site |
| City H | utility-derived, on request | register-anchored, semi-synthetic | derived results only |
| City D | utility-derived, on request | audited 2025 field register (restricted) | derived results only |

The City H and City D network models and the audited leak-to-node mapping are available
from the corresponding authors on reasonable request; the raw register is restricted by
the operating utility. In the released results files, DMA names are replaced by stable
codes and register leak-type labels are translated; **no numeric value is altered**. No
real SCADA pressure data are used anywhere in this study.

## Integrity by construction

- `tests/integrity_lint.py` fails the build if any metric-named variable is assigned a bare
  numeric literal anywhere in production code.
- Acceptance certificates bind each dispatch decision to a SHA-256 digest of the canonical
  evidence serialization; `verify()` detects any post-hoc modification
  (`tests/smoke_certificate.py`).
- Figures and tables are generated exclusively from `artifacts/*.json`; the manuscript
  quotes are diffable via `eval/paper_numbers.py`.
- The LLM auditor sees numbers only, can only add rejections, and defaults to rejection on
  any parse failure.

## Citation

If this work is useful to you, please cite:

```bibtex
@article{mu2026leakagent,
  title   = {Verifiable abstention makes {AI} leak diagnosis accountable in water distribution networks},
  author  = {Mu, Tianwei and Wang, Yue and Yuan, Mingzhe and Huang, Manhong and Wang, Wenhong
             and Yin, Xuerui and Luo, Qing and Xiao, Min and Yang, Hui and Li, Jun and Xue, Dan},
  journal = {arXiv preprint arXiv:2608.18836},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.18836}
}
```

## License

MIT for all code in this repository (see [LICENSE](LICENSE)). The L-Town network model is
redistributed under the terms of the BattLeDIM 2020 competition (KIOS Research and
Innovation Center of Excellence); the KY4 model originates from the Kentucky water
distribution research database as bundled with WNTR. See `data/ltown/ATTRIBUTION.md` and
`data/ky4/ATTRIBUTION.md`.

## Acknowledgements

We thank the two operating utilities for the network models and repair registers behind
the field legs. This work was supported by the National Natural Science Foundation of
China and the Guangdong Basic and Applied Basic Research Foundation (grant numbers in the
paper).
