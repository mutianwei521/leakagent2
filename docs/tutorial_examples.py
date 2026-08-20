# -*- coding: utf-8 -*-
"""Worked (toy) examples for the plain-language tutorial (docs/tutorial_zh.md).

Every number printed here is what the tutorial quotes. The toy network is
illustrative: 3 zones, m = 4 pressure sensors, sigma = 0.05 m. The formulas are
the manuscript's Eq. (2) (residual / Mahalanobis / log-likelihood), Eq. (3)
(Bayesian fusion with the BIC Occam penalty), Eq. (4) (goal contract, EXA7
preset), Supplementary Eq. S6 (active-sensing discriminability), and the
survey-tier gate u*sigma_f/sqrt(W). Run:  python docs/tutorial_examples.py
"""
import math

SIGMA = 0.05
M = 4
PRIOR_FAMILY = {"leak": 0.50, "demand": 0.20, "sensor": 0.15, "valve": 0.10, "null": 0.05}
# EXA7 preset of Eq. (4)
TAU_E, DELTA, ALPHA, R_MAX, RHO_MAX = 0.5, 0.12, 0.20, 60, 3.0


def loglik(obs, mu):
    r = [o - u for o, u in zip(obs, mu)]
    ss = sum(x * x for x in r)
    rho = math.sqrt(ss) / (SIGMA * math.sqrt(M))
    ll = -ss / (2 * SIGMA ** 2) - M * math.log(SIGMA * math.sqrt(2 * math.pi))
    return r, math.sqrt(ss), rho, ll


def fuse(obs, hyps):
    """hyps: list of (name, family, mu, k_free_params, n_in_family)."""
    rows = []
    for name, fam, mu, k, nfam in hyps:
        r, norm, rho, ll = loglik(obs, mu)
        prior = PRIOR_FAMILY[fam] / nfam
        pen = 0.5 * k * math.log(M)
        score = math.log(prior) + ll - pen
        rows.append(dict(name=name, fam=fam, r=r, norm=norm, rho=rho, ll=ll,
                         prior=prior, pen=pen, score=score))
    zmax = max(x["score"] for x in rows)
    Z = sum(math.exp(x["score"] - zmax) for x in rows)
    for x in rows:
        x["post"] = math.exp(x["score"] - zmax) / Z
    rows.sort(key=lambda x: -x["post"])
    return rows


def contract(rows, region_size):
    top, second = rows[0], rows[1]
    pi_exist = sum(x["post"] for x in rows if x["fam"] == "leak")
    margin = top["post"] - second["post"]
    checks = {
        "G1 existence": (top["fam"] == "leak") and pi_exist >= TAU_E,
        "G2 region": region_size <= R_MAX,
        "G3 margin": margin >= DELTA,
        "G4 alternatives": second["post"] <= ALPHA,
        "G6 residual": top["rho"] <= RHO_MAX,
        "G7 safety": True,
    }
    return pi_exist, margin, checks


def show(title, obs, rows, region):
    print("=" * 78)
    print(title)
    print("observed dp:", obs)
    print(f"{'hypothesis':<16}{'||r|| (m)':>10}{'rho':>7}{'logL':>9}{'prior':>8}{'penalty':>9}{'posterior':>11}")
    for x in rows:
        print(f"{x['name']:<16}{x['norm']:>10.4f}{x['rho']:>7.2f}{x['ll']:>9.2f}"
              f"{x['prior']:>8.4f}{x['pen']:>9.3f}{x['post']:>11.4f}")
    pi, mg, ch = contract(rows, region)
    print(f"Pi_exist = {pi:.4f}   margin = {mg:.4f}   top rho = {rows[0]['rho']:.2f}   region = {region}")
    for k, v in ch.items():
        print(f"   {k:<16} {'PASS' if v else 'FAIL'}")
    print("   =>", "ACT" if all(ch.values()) else "do NOT act")
    return pi, mg, ch


# ---------------------------------------------------------------- Example 1
# a clear leak in zone A: sensors 1,2 (inside A) drop hard, sensors 3,4 barely move
obs1 = [-0.42, -0.38, -0.05, -0.02]
hyps1 = [
    ("leak zone A", "leak", [-0.40, -0.35, -0.06, -0.03], 2, 3),
    ("leak zone B", "leak", [-0.10, -0.12, -0.41, -0.36], 2, 3),
    ("leak zone C", "leak", [-0.05, -0.04, -0.08, -0.40], 2, 3),
    ("demand zone A", "demand", [-0.30, -0.30, -0.04, -0.02], 1, 3),
    ("sensor 1 fault", "sensor", [-0.42, 0.00, 0.00, 0.00], 1, 4),
    ("valve V1 closed", "valve", [-0.20, -0.25, -0.15, -0.10], 0, 1),
    ("no anomaly", "null", [0.00, 0.00, 0.00, 0.00], 0, 1),
]
rows1 = fuse(obs1, hyps1)
show("EXAMPLE 1  clear leak in zone A", obs1, rows1, region=18)

# ---------------------------------------------------------------- Example 2
# ambiguous: zones A and B predict similar signatures at the 4 sensors
obs2 = [-0.22, -0.20, -0.21, -0.19]
hyps2 = [
    ("leak zone A", "leak", [-0.24, -0.21, -0.18, -0.17], 2, 3),
    ("leak zone B", "leak", [-0.19, -0.20, -0.24, -0.21], 2, 3),
    ("leak zone C", "leak", [-0.05, -0.04, -0.08, -0.40], 2, 3),
    ("demand zone A", "demand", [-0.15, -0.15, -0.05, -0.04], 1, 3),
    ("sensor 1 fault", "sensor", [-0.22, 0.00, 0.00, 0.00], 1, 4),
    ("valve V1 closed", "valve", [-0.20, -0.25, -0.15, -0.10], 0, 1),
    ("no anomaly", "null", [0.00, 0.00, 0.00, 0.00], 0, 1),
]
rows2 = fuse(obs2, hyps2)
show("EXAMPLE 2  ambiguous between zones A and B", obs2, rows2, region=18)

# active sensing (Supplementary Eq. S6): candidate hidden nodes x1, x2;
# twin-predicted reading under the two surviving hypotheses
print("-" * 78)
print("active sensing: spread of surviving hypotheses' predictions at hidden nodes")
hidden = {"x1": {"leak zone A": -0.55, "leak zone B": -0.05},
          "x2": {"leak zone A": -0.30, "leak zone B": -0.28}}
for x, pred in hidden.items():
    vals = list(pred.values())
    print(f"   {x}: A predicts {pred['leak zone A']:+.2f}, B predicts {pred['leak zone B']:+.2f},"
          f"  spread = {max(vals) - min(vals):.2f} m")
best = max(hidden, key=lambda x: max(hidden[x].values()) - min(hidden[x].values()))
print("   chosen node:", best)
# reveal x1 = -0.52 m and re-fit with m = 5 sensors
obs2b = obs2 + [-0.52]
M = 5
hyps2b = [(n, f, mu + [hidden["x1"].get(n, 0.0) if f == "leak" and n in hidden["x1"] else
                       (mu[0] if f == "sensor" else (-0.10 if f == "valve" else 0.0))],
           k, nf) for (n, f, mu, k, nf) in hyps2]
# leak zone C and demand zone A get a small predicted reading at x1
for h in hyps2b:
    if h[0] == "leak zone C":
        h[2][-1] = -0.06
    if h[0] == "demand zone A":
        h[2][-1] = -0.12
    if h[0] == "sensor 1 fault":
        h[2][-1] = 0.0
rows2b = fuse(obs2b, hyps2b)
show("EXAMPLE 2b  after revealing x1 = -0.52 m (m = 5)", obs2b, rows2b, region=18)
M = 4

# ---------------------------------------------------------------- Example 3
# audit: a corrupted package (unsupported assertion)
print("=" * 78)
print("EXAMPLE 3  auditor: 'unsupported assertion' package")
print("   posterior 0.93 for 'leak zone A' but its rho = 7.8  (>3, physically incompatible)")
print("   -> auditor must reject: high confidence with a residual that does not reproduce the data")

# ---------------------------------------------------------------- Example 4
# survey-tier gate: u * sigma_f / sqrt(W)
print("=" * 78)
print("EXAMPLE 4  survey tier gate  u*sigma_f/sqrt(W)  with u = 3.4, sigma_f = 0.10 L/s")
for W in (1, 5, 14):
    thr = 3.4 * 0.10 / math.sqrt(W)
    print(f"   W = {W:>2} nights: threshold = {thr:.3f} L/s   median register leak 0.106 L/s"
          f" -> {'DETECTED' if 0.106 > thr else 'not detected'}")

# ---------------------------------------------------------------- Example 5
# per-100-alarms cost illustration (from the paper's per-100 numbers)
print("=" * 78)
print("EXAMPLE 5  per 100 alarms (paper, EXA7): forced localizer digs 100, wrong 68;"
      " contract digs 37-38, wrong about 1")
print("   with an ILLUSTRATIVE wrong-dig cost of 10,000 (arbitrary units):"
      f" forced = {68*10000:,}, contract = {1*10000:,}")

# ---------------------------------------------------------------- Example 6
# pressure-information ceiling on City D
print("=" * 78)
print("EXAMPLE 6  City D ceiling: median best clean signature 0.0023 m vs noise 0.15 m")
print(f"   ratio noise/signal = {0.15/0.0023:.0f}x ; signal is {0.0023/0.15*100:.1f}% of one sigma")
