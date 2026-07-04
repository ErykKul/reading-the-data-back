#!/usr/bin/env python3
"""Grow the detection-through-noise pool: harvest a representative sample of political-science / social-science
replication datasets from Harvard Dataverse (tabular + code files only, size-capped), into data/dv/ alongside the
embedded testbed. Resumable (skips datasets already present). This is the noise the verified mismatches must survive.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import dv

DEST = "data/dv"
MAXSZ = 25_000_000          # skip files > 25MB (keeps profiling fast; distribution kind is stable on smaller)
CAP = 400                   # max datasets to add
QUERIES = [
    "replication data political science regression",
    "replication data OLS count outcome",
    "American Political Science Review replication",
    "American Journal of Political Science replication data",
    "Journal of Politics replication data",
    "International Studies Quarterly replication data",
    "Comparative Political Studies replication",
    "conflict casualties replication data regression",
    "vote share replication regression data",
    "public opinion survey replication regression",
    "political economy replication panel regression",
    "civil war onset replication logit count",
]

existing = set(os.listdir(DEST)) if os.path.isdir(DEST) else set()
seen, pool = set(), []
for q in QUERIES:
    try:
        hits = dv.search(q, per_page=120)
    except Exception as e:
        print(f"search fail [{q[:30]}]: {e}", flush=True); continue
    for it in hits:
        gid = it.get("global_id", "")
        if not gid or gid in seen:
            continue
        seen.add(gid)
        dirn = "DVN_" + gid.split("/")[-1]
        if dirn in existing:
            continue
        pool.append((gid, dirn))
    time.sleep(0.3)
print(f"{len(pool)} candidate new datasets (from {len(seen)} unique hits); downloading up to {CAP}", flush=True)

added = 0
for gid, dirn in pool[:CAP]:
    dest = os.path.join(DEST, dirn)
    try:
        files = dv.list_files(gid)
    except Exception:
        continue
    got_tab = got_code = 0
    os.makedirs(dest, exist_ok=True)
    for f in files:
        df = f.get("dataFile", {})
        fn = df.get("filename", ""); fid = df.get("id"); sz = df.get("filesize", 0) or 0
        low = fn.lower()
        is_tab = low.endswith((".tab", ".csv"))
        is_code = low.endswith((".do", ".r", ".rmd"))
        if not (is_tab or is_code) or sz > MAXSZ or fid is None:
            continue
        if is_tab and got_tab >= 6:
            continue
        if is_code and got_code >= 6:
            continue
        try:
            dv.fetch_file(fid, dest=os.path.join(dest, os.path.basename(fn)))
            if is_tab: got_tab += 1
            else: got_code += 1
        except Exception:
            pass
    if got_tab and got_code:
        added += 1
    else:
        # no usable pair; leave the dir (may still profile), but note
        pass
    if added % 15 == 0 and added:
        print(f"  added {added} datasets with tabular+code", flush=True)
    time.sleep(0.2)
print(f"DONE: added {added} datasets with both tabular+code (pool now {len(os.listdir(DEST))} dirs)", flush=True)
