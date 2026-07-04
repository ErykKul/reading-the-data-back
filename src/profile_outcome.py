"""
profile_outcome.py
------------------
Stream a CSV or TAB-delimited file, compute diagnostics on a named outcome
column, and classify it under the paper-2 marginal-distribution taxonomy.

Usage
-----
    python profile_outcome.py <file> <outcome_column>

The script prints one JSON object with all diagnostics and a ``dist_class``
field containing the taxonomy label.  No third-party dependencies; stdlib
only (csv, math, statistics, argparse, json, sys, pathlib).

Taxonomy (marginal class of the modeled outcome column)
-------------------------------------------------------
binary      -- exactly 2 distinct values
ordinal     -- small bounded integer set AND |skew| < 2
categorical -- low-cardinality string (distinct <= 20, non-numeric)
count       -- non-negative integers, many distinct OR |skew| >= 2
proportion  -- all decimal values in [0, 1]
heavy-tailed-- non-negative decimals, high |skew| >= 3 OR orders-of-magnitude
              spread (max/mean >= 100)
continuous  -- everything else numeric

Failure modes (documented in paper, §7):
  - A symmetric small-integer range (e.g. germination 0-5, skew -0.41) is
    indistinguishable from a Likert rating from the marginal alone; this
    profiler will label it ordinal.
  - Scale-masked proportions (e.g. % expressed as 0-100) will be labelled
    continuous rather than proportion.
"""

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys


# ── Streaming accumulators ──────────────────────────────────────────────────

class StreamStats:
    """Welford one-pass mean/variance + running skew (third central moment)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0   # sum of squared deviations
        self._M3 = 0.0   # sum of cubed deviations (for skewness)
        self.min = math.inf
        self.max = -math.inf
        self.n_integer = 0
        self.n_negative = 0
        self.n_in_01 = 0
        self.distinct: set = set()
        self._distinct_overflow = False  # stop tracking after 10 000 values
        self._DISTINCT_CAP = 10_000

    def update(self, x: float, raw: str) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2
        # Welford-style running third central moment
        # Using the stable formula: M3_n = M3_{n-1} + delta*(delta2**2 - M2_{n-1}/n)
        # Simpler recurrence that is numerically fine for research use:
        n = self.n
        if n >= 2:
            self._M3 += (delta * delta2 * (delta * (n - 2) / n)) - (3 * delta2 * self._M2 / n)
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        if x == math.floor(x):
            self.n_integer += 1
        if x < 0:
            self.n_negative += 1
        if 0.0 <= x <= 1.0:
            self.n_in_01 += 1
        if not self._distinct_overflow:
            self.distinct.add(raw)
            if len(self.distinct) > self._DISTINCT_CAP:
                self._distinct_overflow = True

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return self._M2 / (self.n - 1)

    @property
    def skew(self) -> float:
        """Sample skewness (Fisher's moment coefficient)."""
        if self.n < 3:
            return 0.0
        s3 = self.var ** 1.5
        if s3 == 0.0:
            return 0.0
        # population skew = M3/n / sigma^3; adjust to sample: * sqrt(n*(n-1))/(n-2)
        pop_skew = (self._M3 / self.n) / (self.var ** 1.5) if s3 > 0 else 0.0
        n = self.n
        sample_skew = pop_skew * (math.sqrt(n * (n - 1)) / (n - 2))
        return sample_skew

    def distinct_count(self) -> int:
        return len(self.distinct)

    def summary(self) -> dict:
        n = self.n
        return {
            "n": n,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "var": self.var,
            "skew": self.skew,
            "n_integer": self.n_integer,
            "n_negative": self.n_negative,
            "n_in_01": self.n_in_01,
            "distinct": self.distinct_count(),
            "distinct_overflow": self._distinct_overflow,
        }


# ── Distribution classifier ─────────────────────────────────────────────────

def classify(stats: dict, is_string_col: bool) -> str:
    """
    Map streaming diagnostics to the paper-2 taxonomy label.

    Parameters
    ----------
    stats : dict
        Output of StreamStats.summary().
    is_string_col : bool
        True when the column could not be parsed as float for any value
        (or more than 5 % of values were non-numeric).

    Returns
    -------
    str
        One of: binary, ordinal, categorical, count, proportion,
        heavy-tailed, continuous.
    """
    distinct = stats["distinct"]
    n = stats["n"]

    # ── categorical: non-numeric low-cardinality strings ──
    if is_string_col:
        if distinct <= 20:
            return "categorical"
        return "categorical"   # even high-cardinality strings get this label

    # From here all values are numeric.
    skew = stats["skew"]
    mn = stats["min"]
    mx = stats["max"]
    mean = stats["mean"]
    n_neg = stats["n_negative"]
    n_int = stats["n_integer"]
    n_in01 = stats["n_in_01"]

    frac_integer = n_int / n if n else 0.0
    frac_in01 = n_in01 / n if n else 0.0
    is_all_integer = frac_integer > 0.99
    is_non_negative = n_neg == 0

    # ── binary: exactly 2 distinct values ──
    if distinct == 2:
        return "binary"

    # ── ordinal: small bounded integer set, low skew ──
    # "Small bounded" = all integers, non-negative, max <= 20, distinct <= 10
    if (
        is_all_integer
        and is_non_negative
        and mx <= 20
        and distinct <= 10
        and abs(skew) < 2.0
    ):
        return "ordinal"

    # ── count: non-negative integers, many distinct OR high skew ──
    if is_all_integer and is_non_negative:
        return "count"

    # ── proportion: all values in [0, 1] ──
    if frac_in01 > 0.99 and mn >= 0.0 and mx <= 1.0:
        return "proportion"

    # ── heavy-tailed: non-negative decimals with high skew or large spread ──
    if is_non_negative:
        spread = (mx / mean) if mean > 0 else 0.0
        if abs(skew) >= 3.0 or spread >= 100.0:
            return "heavy-tailed"

    # ── continuous: everything else ──
    return "continuous"


# ── Streaming file reader ───────────────────────────────────────────────────

def sniff_delimiter(path: pathlib.Path) -> str:
    """Return ',' or '\\t' based on file suffix; fall back to csv.Sniffer."""
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    if suffix == ".csv":
        return ","
    # Try to sniff from the first line
    with path.open(encoding="utf-8", errors="replace") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        return ","


def stream_column(path: pathlib.Path, col: str, max_rows: int = 500_000):
    """
    Yield (float_value, raw_string, is_numeric) for each non-missing row
    in ``col``.  Raises KeyError if the column is absent.
    """
    delim = sniff_delimiter(path)
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            # Consume the first row to populate fieldnames
            next(reader, None)
        if col not in (reader.fieldnames or []):
            available = ", ".join(reader.fieldnames or [])
            raise KeyError(
                f"Column '{col}' not found. Available columns: {available}"
            )
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            raw = (row[col] or "").strip()
            if not raw or raw.lower() in {"na", "nan", "null", ".", ""}:
                continue
            try:
                yield float(raw), raw, True
            except ValueError:
                yield float("nan"), raw, False


# ── Main ────────────────────────────────────────────────────────────────────

def profile(path: pathlib.Path, col: str, max_rows: int = 500_000) -> dict:
    """
    Profile outcome column ``col`` in ``path``.

    Returns a dict with all streaming diagnostics and a ``dist_class`` key.
    """
    acc = StreamStats()
    n_non_numeric = 0
    n_total = 0

    for fval, raw, is_num in stream_column(path, col, max_rows=max_rows):
        n_total += 1
        if is_num and not math.isnan(fval):
            acc.update(fval, raw)
        else:
            n_non_numeric += 1

    if n_total == 0:
        return {"error": "no data rows found", "dist_class": "unknown"}

    frac_non_numeric = n_non_numeric / n_total
    is_string_col = frac_non_numeric > 0.05 or acc.n == 0

    stats = acc.summary()
    stats["n_non_numeric"] = n_non_numeric
    stats["n_total_rows"] = n_total

    dist_class = classify(stats, is_string_col=is_string_col)
    stats["dist_class"] = dist_class
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile an outcome column in a CSV/TAB file and classify its distribution."
    )
    parser.add_argument("file", help="Path to CSV or TAB file")
    parser.add_argument("column", help="Outcome column name")
    parser.add_argument(
        "--max-rows", type=int, default=500_000,
        help="Maximum rows to stream (default 500 000)"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON output"
    )
    args = parser.parse_args(argv)

    path = pathlib.Path(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = profile(path, args.column, max_rows=args.max_rows)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
