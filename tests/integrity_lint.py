"""
Integrity lint: operationalizes the 'no hard-coded metrics' guarantee by scanning
the production code (agents/ tools/ eval/ schemas/ data/) for suspicious literal
assignments to metric-named variables (e.g. top1 = 0.823). Thresholds, weights,
and seeds are allow-listed. Exit code 1 if anything suspicious is found.

Run:  python tests/integrity_lint.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["agents", "tools", "eval", "schemas", "data", "anomaly_sim", "vendor"]
ROOT_FILES = ["run_diagnosis.py", "reproduce_all.py"]
# metric-named variables that must NOT be assigned a bare float literal
METRIC = re.compile(r"\b(top1|top3|accuracy|acc|precision|recall|f1|mrr|auc|aurc|"
                    r"decision_precision|false_dispatch|catch_rate|ece)\s*=\s*[0-9]+\.[0-9]+")
ALLOW = ("threshold", "tau", "alpha", "beta", "weight", "penalt", "noise", "sigma",
         "grid", "seed", "lr", "_max", "_min", "prob", "margin", "rate")

flagged = []


def scan_file(p):
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        if METRIC.search(s) and not any(a in s.lower() for a in ALLOW):
            flagged.append(f"{os.path.relpath(p, ROOT)}:{i}: {s[:90]}")


for f in ROOT_FILES:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        scan_file(p)
for d in DIRS:
    for root, _, files in os.walk(os.path.join(ROOT, d)):
        if "__pycache__" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            for i, line in enumerate(open(p, encoding="utf-8"), 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if METRIC.search(s) and not any(a in s.lower() for a in ALLOW):
                    flagged.append(f"{os.path.relpath(p, ROOT)}:{i}: {s[:90]}")

print("=" * 56)
print("INTEGRITY LINT — hard-coded-metric scan")
print("=" * 56)
if flagged:
    print(f"FLAGGED {len(flagged)} suspicious literal(s):")
    for f in flagged:
        print("  ", f)
    sys.exit(1)
print("clean: no metric variable is assigned a bare float literal in production code.")
sys.exit(0)
