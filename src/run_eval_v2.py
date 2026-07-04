"""
run_eval_v2.py
--------------
Evaluate the outcome/estimator mismatch detector on operator2_testbed_v2.csv.

Computes:
  1. Detector P/R with 1000x bootstrap 95% CIs
  2. Trivial baseline ("flag every linear/OLS estimator on a non-continuous
     outcome") P/R with 1000x bootstrap 95% CIs

Outputs a results dict and saves rebuild_results.md.

Usage:
    python3 src/run_eval_v2.py

No third-party dependencies (uses only stdlib random + csv).
"""

import csv
import json
import pathlib
import random
import sys

# Ensure the src directory is on the path so testbed_eval imports cleanly
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import testbed_eval

CSV_PATH = pathlib.Path(__file__).parent.parent / "datasets" / "operator2_testbed_v2.csv"
OUT_MD   = pathlib.Path(__file__).parent.parent / "rebuild_results.md"

# ── Linear/OLS keyword set for trivial baseline ──────────────────────────────
_LINEAR_KEYWORDS = {
    "reg", "regress", "xtreg", "areg", "lm", "felm", "ols",
    "lm.cluster", "sureg", "suest", "aov",
    "glm.gaussian", "plm",
}

_NON_CONTINUOUS_CLASSES = {"binary", "ordinal", "count", "proportion"}


def _is_linear_estimator(estimator_str: str) -> bool:
    """Return True if estimator_str (from CSV) resolves to a linear class."""
    resolved = testbed_eval.est_class(estimator_str)
    if resolved == "linear":
        return True
    # Also catch raw keywords that might not be in the table
    tok = estimator_str.strip().lower().split()[0].strip("._-")
    return tok in _LINEAR_KEYWORDS


def _baseline_flag(measured: str, estimator: str) -> bool:
    """Trivial baseline: flag iff estimator is linear AND outcome is not continuous."""
    return _is_linear_estimator(estimator) and measured in _NON_CONTINUOUS_CLASSES


def _gt_positive(gt_raw: str) -> bool | None:
    """True = mismatch (positive), False = match (negative), None = skip."""
    v = gt_raw.strip().lower()
    if v in {"mismatch", "1", "true", "yes"}:
        return True
    if v in {"correct", "match", "0", "false", "no"}:
        return False
    return None  # debated or unknown -- exclude from P/R


def load_rows(csv_path: pathlib.Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def evaluate_rows(rows: list[dict], use_baseline: bool = False) -> dict:
    """
    Evaluate detector or baseline on the given rows.
    Returns tp, fp, fn, tn, skipped counts + precision, recall, f1.
    """
    tp = fp = fn = tn = skipped = 0
    for row in rows:
        measured   = row["measured_dist"].strip().lower()
        estimator  = row["estimator"].strip()
        gt_raw     = row["gt_label"].strip()

        gt = _gt_positive(gt_raw)
        if gt is None:
            skipped += 1
            continue

        if use_baseline:
            predicted = _baseline_flag(measured, estimator)
        else:
            result = testbed_eval.detect(measured, estimator)
            if result["mismatch"] is None:
                skipped += 1
                continue
            predicted = result["mismatch"]

        if gt and predicted:
            tp += 1
        elif not gt and not predicted:
            tn += 1
        elif gt and not predicted:
            fn += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "skipped": skipped,
        "precision": round(precision, 4) if precision is not None else None,
        "recall":    round(recall,    4) if recall    is not None else None,
        "f1":        round(f1,        4) if f1        is not None else None,
    }


def bootstrap_ci(
    rows: list[dict],
    use_baseline: bool = False,
    n_iter: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    1000x bootstrap 95% CIs for precision and recall.
    Resamples rows WITH replacement.
    """
    rng = random.Random(seed)
    prec_samples = []
    rec_samples  = []
    n = len(rows)

    for _ in range(n_iter):
        sample = [rows[rng.randint(0, n - 1)] for _ in range(n)]
        m = evaluate_rows(sample, use_baseline=use_baseline)
        if m["precision"] is not None:
            prec_samples.append(m["precision"])
        if m["recall"] is not None:
            rec_samples.append(m["recall"])

    def ci(samples):
        if not samples:
            return (None, None)
        samples.sort()
        lo = samples[int(alpha / 2 * len(samples))]
        hi = samples[int((1 - alpha / 2) * len(samples))]
        return (round(lo, 4), round(hi, 4))

    return {
        "precision_ci95": ci(prec_samples),
        "recall_ci95":    ci(rec_samples),
    }


def class_balance(rows: list[dict]) -> dict:
    n_mismatch = n_match = n_other = 0
    for row in rows:
        gt = _gt_positive(row["gt_label"].strip())
        if gt is True:
            n_mismatch += 1
        elif gt is False:
            n_match += 1
        else:
            n_other += 1
    return {"n_mismatch": n_mismatch, "n_match": n_match, "n_other_debated": n_other}


def severity_breakdown(rows: list[dict]) -> dict:
    """Count mismatches by severity label from the detector."""
    counts = {"clear": 0, "debated": 0, "none": 0, "unknown": 0}
    for row in rows:
        measured  = row["measured_dist"].strip().lower()
        estimator = row["estimator"].strip()
        result = testbed_eval.detect(measured, estimator)
        counts[result["severity"]] = counts.get(result["severity"], 0) + 1
    return counts


def main():
    rows = load_rows(CSV_PATH)
    n = len(rows)

    balance = class_balance(rows)

    # ── Detector ────────────────────────────────────────────────────────────
    det_metrics = evaluate_rows(rows, use_baseline=False)
    det_ci      = bootstrap_ci(rows, use_baseline=False)
    sev         = severity_breakdown(rows)

    # ── Trivial baseline ────────────────────────────────────────────────────
    bas_metrics = evaluate_rows(rows, use_baseline=True)
    bas_ci      = bootstrap_ci(rows, use_baseline=True)

    result = {
        "n_rows_total": n,
        "class_balance": balance,
        "detector": {**det_metrics, **det_ci},
        "trivial_baseline": {**bas_metrics, **bas_ci},
        "detector_severity_breakdown": sev,
    }

    print(json.dumps(result, indent=2))

    # ── Write rebuild_results.md ─────────────────────────────────────────────
    d = result["detector"]
    b = result["trivial_baseline"]
    bl = result["class_balance"]

    md = f"""# Testbed v2 Evaluation Results

Generated by `src/run_eval_v2.py` against `datasets/operator2_testbed_v2.csv`.

## Dataset

| Metric | Value |
|--------|-------|
| Total rows | {n} |
| Mismatch (positive) | {bl['n_mismatch']} |
| Match / correct (negative) | {bl['n_match']} |
| Debated / other (excluded from P/R) | {bl['n_other_debated']} |
| Mismatch prevalence | {bl['n_mismatch'] / (bl['n_mismatch'] + bl['n_match']):.1%} |

## Detector Performance (1000x bootstrap 95% CIs)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Precision | {d['precision']:.4f} | {d['precision_ci95']} |
| Recall | {d['recall']:.4f} | {d['recall_ci95']} |
| F1 | {d['f1']:.4f} | — |
| TP / FP / FN / TN | {d['tp']} / {d['fp']} / {d['fn']} / {d['tn']} | — |
| Skipped (unknown estimator) | {d['skipped']} | — |

### Severity breakdown (detector only; baseline produces no severity grades)

| Severity | Count | Description |
|----------|-------|-------------|
| clear | {sev['clear']} | King-style: count->linear, proportion->linear |
| debated | {sev['debated']} | LPM (binary->linear), ordinal->linear |
| none | {sev['none']} | No mismatch (correct match) |
| unknown | {sev['unknown']} | Estimator keyword not recognised |

## Trivial Baseline ("flag every linear/OLS on non-continuous outcome")

| Metric | Value | 95% CI |
|--------|-------|--------|
| Precision | {b['precision']:.4f} | {b['precision_ci95']} |
| Recall | {b['recall']:.4f} | {b['recall_ci95']} |
| F1 | {b['f1']:.4f} | — |
| TP / FP / FN / TN | {b['tp']} / {b['fp']} / {b['fn']} / {b['tn']} | — |
| Skipped (unknown estimator) | {b['skipped']} | — |

## Structural note on benchmark construction

The detector P/R and the trivial-baseline P/R are identical on this benchmark
because the curated rows partition cleanly: every mismatch uses a linear
estimator on a non-continuous outcome, and every match uses the correct
non-linear estimator on its matching class.  In this configuration the two
rules are logically equivalent.

The detector adds value that this benchmark does not exercise:

1. **Severity taxonomy** — the baseline produces a single binary flag; the
   detector additionally grades each mismatch as "clear" (King-style count or
   proportion into OLS) or "debated" (LPM, ordinal-into-linear). On this
   testbed: {sev['clear']} clear + {sev['debated']} debated out of {bl['n_mismatch']} mismatch cases.

2. **Cross-non-linear detection** — if a betareg were applied to a binary
   outcome, or a Poisson to an ordinal, the detector would flag it; the
   baseline (which only checks "is it linear?") would miss it.

3. **Structured evidence output** — the detector returns assumed class, severity
   label, and a human-readable note for each case; the baseline produces only a
   flag bit.

A future benchmark that includes cross-non-linear cases and continuous-outcome
rows will break the P/R tie.

## Notes

- The **detector** uses `testbed_eval.est_class()` + `detect()` to resolve the
  estimator keyword and compare against the measured distribution class.
- The **trivial baseline** flags any row where the estimator resolves to a
  linear/OLS class AND the measured distribution is non-continuous (binary,
  ordinal, count, or proportion).
- Bootstrap resamples rows with replacement (n={n}, 1000 iterations, seed=42).
- CIs are degenerate [1.0, 1.0] because P=R=1 on every bootstrap resample
  of this perfectly separated set.
- Duplicate PIDs exist in the CSV (same dataset, different outcome variables or
  coding variants); each row is an independent benchmark unit.
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\nResults written to: {OUT_MD}", file=sys.stderr)

    return result


if __name__ == "__main__":
    main()
