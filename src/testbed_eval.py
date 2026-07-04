"""
testbed_eval.py
---------------
Two public functions:

    est_class(code_text) -> str | None
        Map analysis-code text to the estimator's assumed outcome class by
        scanning for canonical command keywords.  Returns None if no
        recognised estimator is found.

    detect(measured, estimator_or_class) -> dict
        Given the measured distribution class (from profile_outcome) and
        either a free-form code snippet or an already-resolved estimator
        keyword, return a dict::

            {
              "measured":      <str>,
              "assumed":       <str | None>,
              "mismatch":      <bool | None>,   # None = estimator unknown
              "severity":      <"clear" | "debated" | "none" | "unknown">,
              "note":          <str>,
            }

        Severity taxonomy (from §5 / 05_testbed.md):
          clear   -- King-style: count->linear, proportion->linear
          debated -- binary->linear (LPM), ordinal->linear
          none    -- correct (admissible)
          unknown -- estimator could not be resolved

Scoring helper (precision/recall over benchmark CSV)
-----------------------------------------------------

    score_benchmark(csv_path, measured_col, estimator_col, gt_col) -> dict
        Compute precision/recall/F1 for MISMATCH detection over a CSV with
        columns for measured_dist, estimator keyword (or assumed class), and
        gt_label ("mismatch" | "correct" | "debated").  The benchmark file is
        datasets/operator2_testbed.csv.

        gt_label values accepted:
          mismatch | 1 | true  -> positive (a real mismatch)
          correct  | 0 | false -> negative
          debated            -> counted as positive for recall; reported
                                separately in the breakdown.

No third-party dependencies.
"""

import argparse
import csv
import json
import pathlib
import sys
from typing import Optional


# ── Estimator keyword table ─────────────────────────────────────────────────

# Each entry: (list_of_keywords, assumed_class)
# Keywords are matched case-insensitively as whole tokens (word-boundary match
# using a simple split + strip approach — no regex dependency).
_ESTIMATOR_TABLE = [
    # ---- linear / Gaussian ----
    (["regress", "reg", "xtreg", "areg", "lm", "felm",
      "ols", "glm.gaussian", "glm(family=gaussian",
      "sureg", "suest", "aov"], "linear"),
    # ---- count ----
    (["poisson", "nbreg", "glm.nb", "zinb", "zip",
      "negative.binomial", "glm.poisson", "glm(family=poisson",
      "zeroinfl", "menbreg", "mepoisson", "xtpoisson", "xtnbreg"], "count"),
    # ---- binary ----
    (["logit", "probit", "logistic", "glm.binomial",
      "glm(family=binomial", "binomial"], "binary"),
    # ---- ordinal ----
    (["ologit", "oprobit", "polr", "clm", "ordered.logit",
      "ordered.probit"], "ordinal"),
    # ---- proportion / fractional ----
    (["betareg", "fracreg", "fractional", "ordered.beta",
      "glm.beta", "tobit"], "proportion"),
    # ---- survival (out of scope for marginal profiling, but named) ----
    (["stcox", "streg", "coxph", "survreg"], "survival"),
]

# Build a flat lookup: keyword -> class
_KW_TO_CLASS: dict[str, str] = {}
for kw_list, cls in _ESTIMATOR_TABLE:
    for kw in kw_list:
        _KW_TO_CLASS[kw.lower()] = cls


def est_class(code_text: str) -> Optional[str]:
    """
    Scan ``code_text`` (a Stata do-file or R script) for estimator keywords
    and return the assumed outcome distribution class.

    Matching is done by splitting on whitespace and common delimiters and
    checking each token against the keyword table.  The first recognised
    keyword wins (respects typical script ordering where the primary model
    call appears early).

    Returns None if no recognised estimator keyword is found.

    Examples
    --------
    >>> est_class("xtreg ln_bills i.year, fe")
    'linear'
    >>> est_class("nbreg count_bills income pop, vce(robust)")
    'count'
    >>> est_class("logit voted age income")
    'binary'
    """
    # Tokenise: split on whitespace, parens, commas, newlines
    import re
    tokens = re.split(r"[\s,\(\)\n\r\t;]+", code_text.lower())
    for tok in tokens:
        tok = tok.strip("._-")
        if tok in _KW_TO_CLASS:
            return _KW_TO_CLASS[tok]
        # Also check prefixes like "glm.nb" that may appear as "glm" + ".nb"
        for kw, cls in _KW_TO_CLASS.items():
            if tok.startswith(kw):
                return cls
    return None


# ── Admissibility and mismatch logic ────────────────────────────────────────

# Which assumed classes are admissible for each measured class.
# An estimator is admissible if its assumed class is in this set.
_ADMISSIBLE: dict[str, set[str]] = {
    "binary":      {"binary"},
    "ordinal":     {"ordinal", "binary"},   # binary lumping is debated-ok
    "categorical": {"categorical", "binary"},
    "count":       {"count"},
    "proportion":  {"proportion"},
    "heavy-tailed": {"count", "linear", "proportion"},  # ambiguous; no clear ground truth
    "continuous":  {"linear", "heavy-tailed"},
    "survival":    {"survival"},
    "unknown":     set(),
}

# Severity of mismatches: (measured, assumed) -> severity string
_SEVERITY_MAP: dict[tuple[str, str], str] = {
    # King-style clear mismatches
    ("count",       "linear"):     "clear",
    ("proportion",  "linear"):     "clear",
    # Softer / debated
    ("binary",      "linear"):     "debated",   # LPM
    ("ordinal",     "linear"):     "debated",
    ("ordinal",     "binary"):     "debated",
    # Survival modelled as linear
    ("survival",    "linear"):     "clear",
}


def detect(measured: str, estimator_or_class: str) -> dict:
    """
    Detect whether the estimator's assumed class is admissible for the
    measured distribution class.

    Parameters
    ----------
    measured : str
        Distribution class from profile_outcome (e.g. "count").
    estimator_or_class : str
        Either a raw code snippet (will be passed through est_class first),
        or an already-resolved class name (one of the taxonomy labels or
        "linear", "count", etc.).

    Returns
    -------
    dict with keys: measured, assumed, mismatch, severity, note.
    """
    measured = measured.strip().lower()

    # Resolve estimator_or_class: is it already a class name?
    known_classes = set(_ADMISSIBLE.keys()) | {"linear", "survival"}
    if estimator_or_class.strip().lower() in known_classes:
        assumed = estimator_or_class.strip().lower()
    else:
        # Treat as code text
        assumed = est_class(estimator_or_class)

    if assumed is None:
        return {
            "measured": measured,
            "assumed": None,
            "mismatch": None,
            "severity": "unknown",
            "note": "No recognised estimator keyword found in code text.",
        }

    admissible = _ADMISSIBLE.get(measured, set())
    is_mismatch = assumed not in admissible

    if not is_mismatch:
        severity = "none"
        note = f"Estimator '{assumed}' is admissible for '{measured}' outcomes."
    else:
        severity = _SEVERITY_MAP.get((measured, assumed), "clear")
        if severity == "clear":
            note = (
                f"Clear mismatch: '{measured}' outcome modelled with '{assumed}' "
                f"estimator (King-style or GLM misspecification)."
            )
        else:
            note = (
                f"Debated mismatch: '{measured}' outcome modelled with '{assumed}' "
                f"estimator (methodology literature is divided)."
            )

    return {
        "measured": measured,
        "assumed": assumed,
        "mismatch": is_mismatch,
        "severity": severity,
        "note": note,
    }


# ── Benchmark scoring ────────────────────────────────────────────────────────

def _gt_is_positive(gt_val: str) -> Optional[bool]:
    """
    Normalise gt_label to True (positive/mismatch), False (negative/correct),
    or None (debated -- treated separately).
    """
    v = gt_val.strip().lower()
    if v in {"mismatch", "1", "true", "yes"}:
        return True
    if v in {"correct", "match", "0", "false", "no"}:
        return False
    if v in {"debated"}:
        return None   # handled separately
    return None


def score_benchmark(
    csv_path: str,
    measured_col: str = "measured_dist",
    estimator_col: str = "estimator",
    gt_col: str = "gt_label",
) -> dict:
    """
    Compute precision / recall / F1 for mismatch detection over a benchmark CSV.

    The CSV must have columns for:
      - measured_col  : the measured distribution class (e.g. "count")
      - estimator_col : the estimator keyword or assumed class (e.g. "linear")
      - gt_col        : ground truth ("mismatch", "correct", or "debated")

    The function treats "debated" rows as positives for recall purposes and
    reports them separately in the breakdown.

    Returns
    -------
    dict with precision, recall, f1, and a per-row breakdown list.
    """
    path = pathlib.Path(csv_path)
    rows = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "empty CSV"}

    tp = fp = fn = tn = 0
    debated_correct = debated_wrong = 0
    breakdown = []

    for row in rows:
        measured = row.get(measured_col, "").strip().lower()
        estimator = row.get(estimator_col, "").strip()
        gt_raw = row.get(gt_col, "").strip()

        result = detect(measured, estimator)
        predicted_mismatch = result["mismatch"]
        gt_positive = _gt_is_positive(gt_raw)

        entry = {
            "pid": row.get("pid", "?"),
            "measured": measured,
            "estimator": estimator,
            "assumed": result["assumed"],
            "predicted_mismatch": predicted_mismatch,
            "severity": result["severity"],
            "gt_label": gt_raw,
            "correct": None,
        }

        if gt_positive is None:
            # Debated: count separately
            if predicted_mismatch:
                debated_correct += 1
                entry["correct"] = "debated-detected"
            else:
                debated_wrong += 1
                entry["correct"] = "debated-missed"
        elif predicted_mismatch is None:
            entry["correct"] = "estimator-unknown"
        elif gt_positive and predicted_mismatch:
            tp += 1
            entry["correct"] = True
        elif not gt_positive and not predicted_mismatch:
            tn += 1
            entry["correct"] = True
        elif gt_positive and not predicted_mismatch:
            fn += 1
            entry["correct"] = False
        else:
            fp += 1
            entry["correct"] = False

        breakdown.append(entry)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "debated_detected": debated_correct,
        "debated_missed": debated_wrong,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "breakdown": breakdown,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Detect outcome/estimator mismatches and score against a benchmark CSV."
        )
    )
    sub = parser.add_subparsers(dest="cmd")

    # detect sub-command
    det = sub.add_parser("detect", help="Detect a single mismatch.")
    det.add_argument("measured", help="Measured distribution class (e.g. count)")
    det.add_argument("estimator", help="Estimator keyword or class (e.g. linear)")

    # score sub-command
    sc = sub.add_parser("score", help="Score a benchmark CSV.")
    sc.add_argument("csv", help="Path to benchmark CSV")
    sc.add_argument("--measured-col",  default="measured_dist")
    sc.add_argument("--estimator-col", default="estimator")
    sc.add_argument("--gt-col",        default="gt_label")
    sc.add_argument("--pretty",        action="store_true")

    # est_class sub-command
    ec = sub.add_parser("est-class", help="Resolve a code snippet to a class.")
    ec.add_argument("code", help="Code snippet (quoted)")

    args = parser.parse_args(argv)

    if args.cmd == "detect":
        result = detect(args.measured, args.estimator)
        print(json.dumps(result, indent=2))
    elif args.cmd == "score":
        result = score_benchmark(
            args.csv,
            measured_col=args.measured_col,
            estimator_col=args.estimator_col,
            gt_col=args.gt_col,
        )
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent))
    elif args.cmd == "est-class":
        cls = est_class(args.code)
        print(cls if cls else "None")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


# ── Quick self-test (run with: python testbed_eval.py) ──────────────────────
def _self_test():
    assert est_class("xtreg ln_bills i.year, fe") == "linear"
    assert est_class("nbreg count_bills income, vce(robust)") == "count"
    assert est_class("logit voted age income") == "binary"
    assert est_class("ologit satisfaction i.year") == "ordinal"
    assert est_class("betareg turnout income") == "proportion"
    assert est_class("no_estimator_here") is None

    r = detect("count", "linear")
    assert r["mismatch"] is True and r["severity"] == "clear", r

    r = detect("binary", "linear")
    assert r["mismatch"] is True and r["severity"] == "debated", r

    r = detect("count", "count")
    assert r["mismatch"] is False and r["severity"] == "none", r

    r = detect("count", "nbreg count_bills income")
    assert r["mismatch"] is False, r

    print("All self-tests passed.")


if __name__ == "__test__":
    _self_test()
