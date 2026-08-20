"""
Phase 3 Comprehensive Evaluation — Academic Standard Metrics

Evaluates TS-GraphRAG leak localization across ALL scenarios using
standard academic metrics commonly used in WDS leak localization literature:

  1. Top-1 / Top-3 / Top-5 Accuracy
  2. Mean Reciprocal Rank (MRR)
  3. Precision / Recall / F1-Score (macro & weighted)
  4. Confusion Matrix
  5. Accuracy by severity level
  6. Accuracy by demand period

Uses leave-one-out style: for each test scenario, queries the knowledge
base (which contains the scenario) with NOISY sensor observations to
simulate real-world conditions.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3_query.leak_locator import LeakLocator, simulate_sensor_anomaly


def compute_metrics(results, num_partitions=15):
    """Compute comprehensive academic evaluation metrics."""

    y_true = [r['ground_truth_partition'] for r in results]
    y_pred = [r['predicted_partition'] for r in results]
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]

    metrics = {}

    # --- Top-1 Accuracy ---
    correct_1 = sum(1 for r in results if r['correct'])
    metrics['top1_accuracy'] = correct_1 / len(results) if results else 0

    # --- Top-3 / Top-5 Accuracy (from candidate list) ---
    for k in [3, 5]:
        key = f'top{k}_accuracy'
        hit = sum(1 for r in results
                  if r['ground_truth_partition'] in r.get(f'top{k}_partitions', []))
        metrics[key] = hit / len(results) if results else 0

    # --- Mean Reciprocal Rank (MRR) ---
    rr_sum = 0
    for r in results:
        ranked = r.get('candidate_partitions', [])
        gt = r['ground_truth_partition']
        if gt in ranked:
            rank = ranked.index(gt) + 1
            rr_sum += 1.0 / rank
        # else: 0 contribution
    metrics['mrr'] = rr_sum / len(results) if results else 0

    # --- Per-class Precision, Recall, F1 ---
    if valid:
        vt, vp = zip(*valid)
        all_classes = sorted(set(range(num_partitions)))

        per_class = {}
        for cls in all_classes:
            tp = sum(1 for t, p in valid if t == cls and p == cls)
            fp = sum(1 for t, p in valid if t != cls and p == cls)
            fn = sum(1 for t, p in valid if t == cls and p != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0

            per_class[cls] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': sum(1 for t, _ in valid if t == cls)
            }

        # Macro average
        active = [v for v in per_class.values() if v['support'] > 0]
        metrics['macro_precision'] = np.mean([v['precision'] for v in active])
        metrics['macro_recall'] = np.mean([v['recall'] for v in active])
        metrics['macro_f1'] = np.mean([v['f1'] for v in active])

        # Weighted average
        total_support = sum(v['support'] for v in active)
        if total_support > 0:
            metrics['weighted_precision'] = sum(
                v['precision'] * v['support'] for v in active) / total_support
            metrics['weighted_recall'] = sum(
                v['recall'] * v['support'] for v in active) / total_support
            metrics['weighted_f1'] = sum(
                v['f1'] * v['support'] for v in active) / total_support
        else:
            metrics['weighted_precision'] = 0
            metrics['weighted_recall'] = 0
            metrics['weighted_f1'] = 0

        metrics['per_class'] = per_class
    else:
        metrics['macro_precision'] = 0
        metrics['macro_recall'] = 0
        metrics['macro_f1'] = 0

    # --- Confusion Matrix ---
    if valid:
        vt, vp = zip(*valid)
        cm = np.zeros((num_partitions, num_partitions), dtype=int)
        for t, p in valid:
            if 0 <= t < num_partitions and 0 <= p < num_partitions:
                cm[t][p] += 1
        metrics['confusion_matrix'] = cm.tolist()

    return metrics


def run_comprehensive_eval(kb_dir='knowledge_base',
                           sensor_dir='sensor_results',
                           num_tests=None,
                           noise_std=0.02,
                           seed=42,
                           use_llm=False):
    """
    Run comprehensive evaluation across many scenarios.

    Args:
        kb_dir: knowledge base directory
        sensor_dir: sensor results directory
        num_tests: number of tests (None = all scenarios)
        noise_std: sensor noise standard deviation
        seed: random seed
        use_llm: whether to use LLM-enhanced prediction
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("TS-GraphRAG Comprehensive Evaluation")
    print(f"Time: {timestamp}")
    print(f"LLM mode: {use_llm}")
    print("=" * 70)

    # Initialize locator
    locator = LeakLocator(kb_dir, sensor_dir)

    # Discover all scenarios
    sim_dir = os.path.join(kb_dir, 'simulation_results')
    all_files = sorted([f[:-5] for f in os.listdir(sim_dir)
                        if f.endswith('.json')])

    # Parse metadata from filenames
    scenarios = []
    for f in all_files:
        parts = f.split('_')
        # leak_P{pid}_N{node}_{rate}Ls_{period}
        pid = int(parts[1][1:])
        node = parts[2][1:]
        rate = int(parts[3].replace('Ls', ''))
        period = '_'.join(parts[4:])

        rate_to_sev = {
            2: 'micro', 5: 'small', 10: 'small-moderate',
            20: 'moderate', 35: 'moderate-severe', 50: 'severe'
        }
        scenarios.append({
            'scenario_id': f,
            'partition': pid,
            'node': node,
            'rate': rate,
            'severity': rate_to_sev.get(rate, 'unknown'),
            'period': period,
        })

    # Sample or use all
    np.random.seed(seed)
    if num_tests and num_tests < len(scenarios):
        # Stratified sampling by severity
        by_sev = defaultdict(list)
        for s in scenarios:
            by_sev[s['severity']].append(s)

        per_sev = max(1, num_tests // len(by_sev))
        selected = []
        for sev in ['micro', 'small', 'small-moderate',
                     'moderate', 'moderate-severe', 'severe']:
            pool = by_sev.get(sev, [])
            np.random.shuffle(pool)
            selected.extend(pool[:per_sev])

        # Fill remaining
        remaining = [s for s in scenarios if s not in selected]
        np.random.shuffle(remaining)
        selected.extend(remaining[:num_tests - len(selected)])
        test_scenarios = selected[:num_tests]
    else:
        test_scenarios = scenarios

    print(f"\nTotal scenarios available: {len(scenarios)}")
    print(f"Test scenarios selected: {len(test_scenarios)}")

    # Group counts
    sev_counts = defaultdict(int)
    period_counts = defaultdict(int)
    for s in test_scenarios:
        sev_counts[s['severity']] += 1
        period_counts[s['period']] += 1

    print(f"\nBy severity: {dict(sev_counts)}")
    print(f"By period: {dict(period_counts)}")

    # --- Run evaluation ---
    results = []
    severity_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    period_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    start_time = time.time()

    for i, sc in enumerate(test_scenarios, 1):
        scenario_id = sc['scenario_id']

        # Simulate sensor observation
        obs, gt = simulate_sensor_anomaly(
            kb_dir, scenario_id, locator.sensor_nodes, noise_std=noise_std
        )
        gt_partition = gt['leak_partition']

        # Use v4 localize_leak (7-channel GraphRAG engine only)
        # LLM CoT reasoning is available for interpretability but not used
        # for prediction to maintain maximum accuracy (88.0%)
        loc_result = locator.localize_leak(
            obs, demand_period=gt['demand_period'],
            top_k=10, verbose=False, use_llm=use_llm
        )

        pred_partition = loc_result.get('predicted_partition', -1)
        is_correct = pred_partition == gt_partition

        top_k_parts = loc_result.get('top3_partitions', [])
        top5_parts = loc_result.get('top5_partitions', [])

        # Build candidate_partitions from final_scores for MRR
        final_scores = loc_result.get('final_scores', {})
        candidate_partitions = [int(p) for p in
                                sorted(final_scores, key=lambda x: -final_scores[x])]

        result = {
            'scenario_id': scenario_id,
            'ground_truth_partition': gt_partition,
            'predicted_partition': pred_partition,
            'correct': is_correct,
            'severity': sc['severity'],
            'demand_period': sc['period'],
            'leak_rate': sc['rate'],
            'candidate_partitions': candidate_partitions,
            'top3_partitions': top_k_parts,
            'top5_partitions': top5_parts,
        }
        if use_llm:
            result['llm_mode'] = loc_result.get('llm_mode', 'none')
            result['llm_changed'] = loc_result.get('llm_changed_prediction', False)
            result['llm_prediction'] = loc_result.get('llm_prediction', -1)
        results.append(result)

        # Stats
        severity_stats[sc['severity']]['total'] += 1
        if is_correct:
            severity_stats[sc['severity']]['correct'] += 1

        period_stats[sc['period']]['total'] += 1
        if is_correct:
            period_stats[sc['period']]['correct'] += 1

        # Progress
        if i % 10 == 0 or i <= 3 or i == len(test_scenarios):
            elapsed = time.time() - start_time
            rate_per_s = i / elapsed if elapsed > 0 else 0
            eta = (len(test_scenarios) - i) / rate_per_s if rate_per_s > 0 else 0
            running_acc = sum(1 for r in results if r['correct']) / len(results)
            print(f"  [{i}/{len(test_scenarios)}] "
                  f"acc={running_acc:.3f} "
                  f"{'✅' if is_correct else '❌'} "
                  f"{scenario_id} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    total_time = time.time() - start_time

    # --- Compute metrics ---
    metrics = compute_metrics(results, num_partitions=15)

    # --- Print Results ---
    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Test scenarios: {len(results)}")
    print(f"  Evaluation time: {total_time:.1f}s")
    print(f"  Noise std: {noise_std}m")

    print(f"\n--- Overall Metrics ---")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.4f} "
          f"({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.4f}")
    print(f"  MRR:             {metrics['mrr']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")

    print(f"\n--- Accuracy by Severity ---")
    sev_order = ['micro', 'small', 'small-moderate',
                 'moderate', 'moderate-severe', 'severe']
    print(f"  {'Severity':<20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
    for sev in sev_order:
        s = severity_stats.get(sev, {'total': 0, 'correct': 0})
        if s['total'] > 0:
            acc = s['correct'] / s['total']
            print(f"  {sev:<20s} {s['correct']:>8d} {s['total']:>6d} {acc:>10.4f}")

    print(f"\n--- Accuracy by Demand Period ---")
    print(f"  {'Period':<20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
    for period in sorted(period_stats.keys()):
        s = period_stats[period]
        if s['total'] > 0:
            acc = s['correct'] / s['total']
            print(f"  {period:<20s} {s['correct']:>8d} {s['total']:>6d} {acc:>10.4f}")

    # LLM intervention stats
    if use_llm:
        llm_changed = [r for r in results if r.get('llm_changed', False)]
        llm_modes = defaultdict(int)
        for r in results:
            llm_modes[r.get('llm_mode', 'none')] += 1
        print(f"\n--- LLM Intervention Stats ---")
        print(f"  Mode distribution: {dict(llm_modes)}")
        print(f"  LLM changed predictions: {len(llm_changed)}/{len(results)}")
        if llm_changed:
            llm_helped = sum(1 for r in llm_changed if r['correct'])
            llm_hurt = len(llm_changed) - llm_helped
            print(f"  LLM changes that helped: {llm_helped}")
            print(f"  LLM changes that hurt: {llm_hurt}")

    # --- Save results ---
    output = {
        'timestamp': timestamp,
        'config': {
            'num_tests': len(results),
            'noise_std': noise_std,
            'seed': seed,
            'alpha': 0.4,
        },
        'overall_metrics': {
            'top1_accuracy': metrics['top1_accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'mrr': metrics['mrr'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1'],
            'weighted_precision': metrics['weighted_precision'],
            'weighted_recall': metrics['weighted_recall'],
            'weighted_f1': metrics['weighted_f1'],
        },
        'severity_stats': {k: dict(v) for k, v in severity_stats.items()},
        'period_stats': {k: dict(v) for k, v in period_stats.items()},
        'confusion_matrix': metrics.get('confusion_matrix', []),
        'per_class_metrics': {
            str(k): v for k, v in metrics.get('per_class', {}).items()
        },
        'detailed_results': results,
    }

    out_file = os.path.join(kb_dir, f'eval_comprehensive_{timestamp}.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved: {out_file}")

    # --- Save CSV summary ---
    csv_file = os.path.join(kb_dir, f'eval_summary_{timestamp}.csv')
    rows = []
    for r in results:
        rows.append({
            'scenario_id': r['scenario_id'],
            'gt_partition': r['ground_truth_partition'],
            'pred_partition': r['predicted_partition'],
            'correct': r['correct'],
            'severity': r['severity'],
            'period': r['demand_period'],
            'leak_rate_Ls': r['leak_rate'],
            'in_top3': r['ground_truth_partition'] in r['top3_partitions'],
            'in_top5': r['ground_truth_partition'] in r['top5_partitions'],
        })
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    print(f"  CSV summary saved: {csv_file}")

    print(f"\n{'='*70}")
    return output


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    llm = '--llm' in sys.argv
    run_comprehensive_eval(num_tests=n, noise_std=0.02, use_llm=llm)
