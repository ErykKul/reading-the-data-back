#!/usr/bin/env python3
"""Compute the detection-through-noise measurements from atscale_flags.json + the testbed labels.
Outputs: recovery (x/32, y/8), outcome-identification accuracy on the testbed, corpus counts, flag volume/rate
per tier, and the list of NOVEL (non-testbed) clear-tier flags for adjudication."""
import sys, json, os
sys.path.insert(0, "src")
import pandas as pd
from atscale_pipeline import profile_dataset, extract_regressions, assumed_kind, NONCONT

res = json.load(open("atscale_flags.json"))
tb = pd.read_csv("datasets/operator2_testbed_v2.csv")
tb["d"] = "DVN_" + tb.pid.str.split("/").str[-1]

# ---- recovery (case-level over the 40 testbed rows, and dataset-level) ----
in_pool = set(res)
tb["in_pool"] = tb.d.isin(in_pool)
mm = tb[tb.gt_label == "mismatch"]; ok = tb[tb.gt_label == "match"]
def flagged_outcome(d, ov):
    return any(f["outcome"] == ov.lower() for f in res.get(d, {}).get("flags", []))
def flagged_any(d):
    return len(res.get(d, {}).get("flags", [])) > 0
mm_recovered = sum(flagged_any(r.d) for r in mm.itertuples())   # dataset-level: its mismatch flagged
mm_outcome = sum(flagged_outcome(r.d, r.outcome_var) for r in mm.itertuples())  # exact outcome flagged
ok_passed = sum(not flagged_any(r.d) for r in ok.itertuples())
print(f"[RECOVERY] testbed in pool: {tb.in_pool.sum()}/{len(tb)} rows ({tb[tb.in_pool].d.nunique()} datasets)")
print(f"  mismatches recovered (dataset flagged): {mm_recovered}/{len(mm)}")
print(f"  mismatches recovered (exact outcome flagged): {mm_outcome}/{len(mm)}")
print(f"  controls passed (no flag): {ok_passed}/{len(ok)}")

# ---- outcome-identification accuracy on the testbed (was the true outcome profiled + correctly classed?) ----
# Needs the raw pool: profiles the deposited data live (rebuild with `make corpus`).
if not os.path.isdir("data/dv"):
    print("\n[OUTCOME-ID] SKIPPED (data/dv not present; needs the raw pool)")
else:
    loc = cls = n = 0
    miss_taxonomy = {"not_in_data": 0, "misclassified": 0, "profiled_ok": 0}
    for r in tb.itertuples():
        if not r.in_pool:
            continue
        n += 1
        kinds = profile_dataset(f"data/dv/{r.d}")
        k = kinds.get(r.outcome_var.lower())
        if k is None:
            miss_taxonomy["not_in_data"] += 1
        else:
            loc += 1
            if r.gt_label == "mismatch" and k in NONCONT:
                cls += 1; miss_taxonomy["profiled_ok"] += 1
            elif r.gt_label == "match":
                cls += 1
            else:
                miss_taxonomy["misclassified"] += 1
    print(f"\n[OUTCOME-ID] on {n} in-pool testbed cases: outcome located in data {loc}/{n} ({100*loc/n:.0f}%)")
    print(f"  miss taxonomy: {miss_taxonomy}")

# ---- corpus + flag volume ----
tot = sum(len(x["flags"]) for x in res.values())
fds = sum(1 for x in res.values() if x["flags"])
clear = sum(1 for x in res.values() for f in x["flags"] if f["severity"] == "clear")
deb = tot - clear
kinds_tot = sum(x["kinds_n"] for x in res.values())
print(f"\n[CORPUS] {len(res)} datasets profiled | {kinds_tot} variables | {fds} flagged datasets | {tot} flags "
      f"({clear} clear, {deb} debated) | flag rate {1000*fds/len(res):.0f}/1000 datasets")

# ---- novel (non-testbed) clear-tier flags for adjudication ----
tb_dirs = set(tb.d)
novel = []
for d, x in res.items():
    if d in tb_dirs:
        continue
    for f in x["flags"]:
        if f["severity"] == "clear":
            novel.append({"dataset": d, **f})
print(f"\n[NOVEL] {len(novel)} clear-tier flags outside the testbed, across {len({n['dataset'] for n in novel})} datasets")
json.dump(novel, open("atscale_novel_flags.json", "w"), indent=0)
# by kind
from collections import Counter
print("  novel clear flags by measured kind:", dict(Counter(n["kind"] for n in novel)))
print("  sample:", [(n["dataset"], n["estimator"], n["outcome"], n["kind"]) for n in novel[:8]])
