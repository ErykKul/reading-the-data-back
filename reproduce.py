#!/usr/bin/env python3
"""Regenerate EVERY number paper2.tex cites, from the committed artifacts (+ raw data when present).

One command, clearly labeled output, no hardcoded results: this is the source of truth the prose must
match, so a non-reproducing number (the round-3 kappa failure) cannot recur. Numbers that need the raw
20GB pool (data/dv) are recomputed when it is present and marked SKIPPED otherwise; everything else
recomputes from the committed JSON/CSV artifacts alone.

Usage:  python reproduce.py   (from the package root; src/ is added to the path automatically)

Inputs (committed): atscale_flags.json, datasets/operator2_testbed_v2.csv,
                    atscale_census_verdicts.json, atscale_refit_broken.json
Inputs (optional):  data/dv/ (raw census pool) for outcome-ID accuracy + the re-analysis anchor.
"""
import json, math, os, sys
from collections import Counter

sys.path.insert(0, "src")
import pandas as pd

HAVE_DATA = os.path.isdir("data/dv")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


def pct(k, n):
    return f"{k}/{n} = {100*k/n:.0f}%"


print("=" * 88)
print("PAPER 2 NUMBER REGENERATION (census edition)".center(88))
print("=" * 88)

res = json.load(open("atscale_flags.json"))
tb = pd.read_csv("datasets/operator2_testbed_v2.csv")
tb["d"] = "DVN_" + tb.pid.str.split("/").str[-1]

# ---------------------------------------------------------------- corpus + denominators
n = len(res)
vars_tot = sum(x["kinds_n"] for x in res.values())
zero = sum(1 for x in res.values() if x["regs_n"] == 0)
nomatch = sum(1 for x in res.values() if x["regs_n"] > 0 and x.get("matched_outcomes", 0) == 0)
flaggable = sum(1 for x in res.values() if x.get("matched_outcomes", 0) >= 1)
print(f"""
[CORPUS]  census of the political-science replication tier
  datasets profiled:            {n:,}
  variables profiled:           {vars_tot:,}

[DENOMINATORS]
  zero extractable regressions: {pct(zero, n)}   (unflaggable by construction)
  regressions, no matched col:  {pct(nomatch, n)}
  flaggable base (>=1 matched): {pct(flaggable, n)}""")

# ---------------------------------------------------------------- flag volume
fds = sum(1 for x in res.values() if x["flags"])
tot = sum(len(x["flags"]) for x in res.values())
clear = sum(1 for x in res.values() for f in x["flags"] if f["severity"] == "clear")
print(f"""
[FLAG VOLUME]
  flagged datasets:             {fds}   ({1000*fds/n:.0f} per 1,000 raw; {100*fds/flaggable:.0f}% of flaggable base)
  flags total:                  {tot:,}   ({clear} clear-tier, {tot-clear} debated)""")

# ---------------------------------------------------------------- testbed + the disclosed tie
from atscale_pipeline import assumed_kind, NONCONT
mm, ok = tb[tb.gt_label == "mismatch"], tb[tb.gt_label == "match"]
tp = sum(1 for r in mm.itertuples()
         if assumed_kind(r.estimator.lower()) == "continuous" and r.measured_dist in NONCONT)
fp = sum(1 for r in ok.itertuples()
         if assumed_kind(r.estimator.lower()) == "continuous" and r.measured_dist in NONCONT)
print(f"""
[TESTBED]  {len(tb)} cases ({len(mm)} mismatch, {len(ok)} controls), {tb.d.nunique()} datasets
  detector on recorded classes: P = {tp/(tp+fp):.1f}, R = {tp/len(mm):.1f}  (== one-line baseline; the disclosed tie)""")

# ---------------------------------------------------------------- recovery through noise
in_pool = set(res)
mmd, okd = mm[mm.d.isin(in_pool)], ok[ok.d.isin(in_pool)]
def flagged_any(d):
    return len(res.get(d, {}).get("flags", [])) > 0
def flagged_outcome(d, ov):
    return any(f["outcome"] == ov.lower() for f in res.get(d, {}).get("flags", []))
ds_rec = sum(flagged_any(d) for d in mmd.d.unique())
ctl_pass = sum(not flagged_any(d) for d in okd.d.unique())
row_rec = sum(flagged_any(r.d) for r in mmd.itertuples())
exact = sum(flagged_outcome(r.d, r.outcome_var) for r in mmd.itertuples())
print(f"""
[RECOVERY]  (dataset-level primary; in-census qualifiers on both sides)
  mismatch datasets in census:  {mmd.d.nunique()}/{mm.d.nunique()}   -> recovered {pct(ds_rec, mmd.d.nunique())}  ({ds_rec}/{mm.d.nunique()} overall)
  control  datasets in census:  {okd.d.nunique()}/{ok.d.nunique()}    -> passed    {ctl_pass}/{okd.d.nunique()}
  secondary, case-level:        {row_rec}/{len(mmd)} rows in a flagged dataset; {exact}/{len(mmd)} exact outcome flagged""")

# ---------------------------------------------------------------- outcome-ID accuracy (needs raw data)
if HAVE_DATA:
    from atscale_pipeline import profile_dataset
    loc = nn = 0
    for r in tb[tb.d.isin(in_pool)].itertuples():
        nn += 1
        kinds = profile_dataset(f"data/dv/{r.d}")
        if kinds.get(r.outcome_var.lower()) is not None:
            loc += 1
    print(f"""
[OUTCOME-ID]  true outcome located in profiled data (in-census testbed cases): {pct(loc, nn)}""")
else:
    print("\n[OUTCOME-ID]  SKIPPED (data/dv not present; needs the raw pool)")

# ---------------------------------------------------------------- novel clear-tier flags
tbd = set(tb.d)
novel = [f for d, x in res.items() if d not in tbd for f in x["flags"] if f["severity"] == "clear"]
kc = Counter(f["kind"] for f in novel)
print(f"""
[NOVEL CLEAR-TIER FLAGS]  outside the testbed
  total: {len(novel)}   (proportion {kc['proportion']}, count {kc['count']})""")

# ---------------------------------------------------------------- two-rater precision + kappa (honest dedup)
v = json.load(open("atscale_census_verdicts.json"))
A, B = v["raterA"], v["raterB"]
key = lambda x: (x["dataset"], x["outcome"])

uA, contraA = {}, set()
for x in A:
    k = key(x)
    if k in uA and uA[k]["real"] != x["real"]:
        contraA.add(k)
    uA.setdefault(k, x)
uB, contraB = {}, set()
for x in B:
    k = key(x)
    if k in uB and uB[k]["real"] != x["real"]:
        contraB.add(k)
    uB.setdefault(k, x)
assert not contraA, f"rater A self-contradictions: {contraA}"

overall = sum(1 for x in uA.values() if x["real"])
lo, hi = wilson(overall, len(uA))
print(f"""
[PRECISION]  two-rater blind adjudication of the balanced novel-flag sample
  sampled flag records:         {len(A)} rater-A + {len(B)} rater-B  (one sampling duplicate disclosed below)
  UNIQUE questions (rater A):   {len(uA)}
  overall precision:            {pct(overall, len(uA))}   Wilson 95% CI [{lo:.2f}, {hi:.2f}]""")
for kind in ("proportion", "count"):
    xs = [x for x in uA.values() if x["kind"] == kind]
    k = sum(1 for x in xs if x["real"])
    lo, hi = wilson(k, len(xs))
    print(f"  {kind:<12} precision:       {pct(k, len(xs))}   CI [{lo:.2f}, {hi:.2f}]")
rb = sum(1 for x in B if x["real"])
print(f"  rater B independent estimate: {pct(rb, len(B))} (raw records)")

# kappa on unique overlap items; a rater self-contradiction on a unique item counts as DISAGREEMENT
pairs = []
for k, xb in uB.items():
    if k not in uA:
        continue
    a = uA[k]["real"]
    b = (not a) if k in contraB else xb["real"]
    pairs.append((a, b))
np_ = len(pairs)
agree = sum(1 for a, b in pairs if a == b)
pa = agree / np_
ay = sum(1 for a, _ in pairs if a) / np_
by = sum(1 for _, b in pairs if b) / np_
pe = ay * by + (1 - ay) * (1 - by)
kappa = (pa - pe) / (1 - pe)
# the favorable collapse (round-3 error): dedup B by last verdict, hiding the self-contradiction
pairs2 = [(uA[k]["real"], [x["real"] for x in B if key(x) == k][-1]) for k in uB if k in uA]
ag2 = sum(1 for a, b in pairs2 if a == b) / len(pairs2)
ay2 = sum(1 for a, _ in pairs2 if a) / len(pairs2)
by2 = sum(1 for _, b in pairs2 if b) / len(pairs2)
pe2 = ay2 * by2 + (1 - ay2) * (1 - by2)
kap2 = (ag2 - pe2) / (1 - pe2)
print(f"""
[INTER-RATER RELIABILITY]  unique overlap items: {np_}
  agreement:                    {agree}/{np_} = {100*pa:.1f}%
  Cohen's kappa (HONEST):       {kappa:.2f}   <- report ~0.90; the self-contradicting sampling duplicate
                                        (DVN_N2GGZ1, adjudicated True AND False by rater B) counts as a
                                        disagreement; items at issue: real disagreement DVN_FCZDKD + the
                                        N2GGZ1 self-contradiction
  favorable collapse (DO NOT USE): {kap2:.2f}  <- the round-3-caught error (last-verdict dedup)""")

# ---------------------------------------------------------------- re-analysis anchor (needs raw data)
if HAVE_DATA:
    from refit_broken import dataset_specs, find_file, load_table, refit_one, term_vars
    specs = [s for s in dataset_specs("data/dv/DVN_HRK5HI") if s[1] == "hcam1" and isinstance(s[2], list)]
    est, dep, terms = specs[0]
    df = load_table(find_file("data/dv/DVN_HRK5HI", {dep} | term_vars(terms)))
    r = refit_one(df, dep, terms, "count")
    print(f"""
[RE-ANALYSIS ANCHOR]  hate crimes (DVN_HRK5HI), deposited headline spec refit as OLS
  fitted values < 0:            {100*r['viol_share']:.0f}% of {r['n']} observations (the published 29% case)""")
else:
    print("\n[RE-ANALYSIS ANCHOR]  SKIPPED (data/dv not present)")

# ---------------------------------------------------------------- provably-broken refit experiment
rb = json.load(open("atscale_refit_broken.json"))
flags = rb["flags"]
okr = [r for r in flags if r["status"] == "refit_ok"]
tax = Counter(r["status"] for r in flags)
b_any = [r for r in okr if r["viol_share"] > 0]
b1 = [r for r in okr if r["viol_share"] >= 0.01]
b5 = [r for r in okr if r["viol_share"] >= 0.05]
vs = sorted(r["viol_share"] for r in b1)
med = vs[len(vs) // 2] if vs else float("nan")
print(f"""
[PROVABLY BROKEN AS A MODEL]  automated OLS refit of deposited clear-tier specifications (no labeling)
  clear-tier flags assessed:    {len(flags)}   (across {len({r['dataset'] for r in flags})} datasets)
  faithfully refittable:        {pct(len(okr), len(flags))}   (plain reg/lm specs; FE/IV/mixed excluded by design)
  coverage taxonomy:            {dict(sorted(tax.items(), key=lambda t: -t[1]))}
  out-of-range fitted values:
    any observation:            {pct(len(b_any), len(okr))} of refittable
    >= 1% of rows (PRIMARY):    {pct(len(b1), len(okr))}   in {len({r['dataset'] for r in b1})} datasets
    >= 5% of rows:              {pct(len(b5), len(okr))}
  violating share among broken (>=1%): median {100*med:.0f}%, max {100*max(vs):.0f}%""")
for kind in ("proportion", "count"):
    kk = [r for r in okr if r["kind"] == kind]
    kb = [r for r in kk if r["viol_share"] >= 0.01]
    print(f"    {kind:<12} broken >=1%:   {pct(len(kb), len(kk))}")
chk = rb["summary"]
assert chk["refit_ok"] == len(okr) and chk["broken_ge1pct"] == len(b1), "stored refit summary drifted"
print(f"  CHECK: stored summary matches recomputation: OK")

print("\n" + "=" * 88)
print("every figure above recomputes from the committed artifacts; prose must match THIS output".center(88))
print("=" * 88)
