#!/usr/bin/env python3
"""Prep the census-scale adjudication sample once atscale_novel_flags.json exists (written by the census
pipeline). Stratifies a properly-powered sample across the clean-tier classes now that counts are no longer
n=5, and pre-marks a two-rater IRR subsample so inter-rater reliability is built in, not bolted on.

The refined classifier already routes single-item rescaled Likerts to ordinal (debated tier), so the clear-tier
proportion flags here should be dominated by native shares/rates (the rater-robust class). Run AFTER the census
pipeline: PYTHONPATH=src python src/prep_census_adjudication.py

BUG FIXED 2026-07-03 (round-3 kappa lesson): the flag pool can carry SEVERAL flags for the same
(dataset, outcome), one per deposited estimator (e.g. reg and xtreg on the same DV). The adjudication
question ("is this outcome genuinely a count/proportion?") is per (dataset, outcome), so sampling raw
flags can put the SAME question in the sample twice; that is exactly how DVN_N2GGZ1 entered the round-2
sample twice, rater B answered it inconsistently, and a favorable dedup inflated kappa 0.90 -> 0.95.
The pool is now deduplicated to unique (dataset, outcome) questions BEFORE sampling (estimators kept as
a list on the item), and the script refuses to overwrite an existing adjudicated sample without --force,
because regenerating the sample invalidates any verdicts already collected on it.
"""
import json, random, os, sys
random.seed(41)

FORCE = "--force" in sys.argv
for out in ("atscale_census_sample.json", "atscale_census_irr.json"):
    if os.path.exists(out) and not FORCE:
        sys.exit(f"{out} exists; it may already be adjudicated. Re-run with --force only if you intend "
                 "to regenerate the sample and re-adjudicate from scratch.")

novel = json.load(open("atscale_novel_flags.json"))

# Dedup to unique adjudication questions: one item per (dataset, outcome), estimators aggregated.
uniq = {}
for f in novel:
    k = (f["dataset"], f["outcome"])
    if k in uniq:
        if f["estimator"] not in uniq[k]["estimators"]:
            uniq[k]["estimators"].append(f["estimator"])
    else:
        uniq[k] = {"dataset": f["dataset"], "outcome": f["outcome"], "kind": f["kind"],
                   "severity": f["severity"], "estimators": [f["estimator"]]}
pool = list(uniq.values())
counts = [n for n in pool if n["kind"] == "count"]
props = [n for n in pool if n["kind"] == "proportion"]
print(f"census clear-tier novel pool: {len(novel)} flags -> {len(pool)} unique (dataset, outcome) questions "
      f"({len(props)} proportion, {len(counts)} count)")

# Powered stratified sample: up to 60 proportion + up to 60 count -> ~120, enough for per-class CIs
n_prop = min(60, len(props))
n_count = min(60, len(counts))
sample = random.sample(props, n_prop) + random.sample(counts, n_count)
random.shuffle(sample)
json.dump(sample, open("atscale_census_sample.json", "w"), indent=0)
print(f"stratified adjudication sample: {n_prop} proportion + {n_count} count = {len(sample)} (all unique)")

# IRR subsample: a random 40 of the sample get a blind second rater -> kappa
irr = random.sample(sample, min(40, len(sample)))
json.dump(irr, open("atscale_census_irr.json", "w"), indent=0)
print(f"IRR subsample (blind second rater): {len(irr)} unique questions")
print(f"class balance now supports per-class precision CIs: proportion n={n_prop}, count n={n_count}")
