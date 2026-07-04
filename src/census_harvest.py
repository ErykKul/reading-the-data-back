#!/usr/bin/env python3
"""CENSUS the political-science replication tier: harvest every dataset in the six leading journal
collections (a census of the population the detector is meant to audit, not a convenience sample), with
tabular + code files (size-capped). Resumable: skips datasets already on disk. This replaces the 486-dataset
sample so 'repository scale' is a real claim about a real population.
"""
import sys, os, time, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import dv

DEST = "data/dv"
MAXSZ = 25_000_000
SUBTREES = ["ajps", "jop", "the_review", "pan", "isq", "cps"]  # AJPS, JoP, APSR, Political Analysis, ISQ, CPS


def search_page(subtree, start, per=100):
    p = {"q": "*", "type": "dataset", "subtree": subtree, "per_page": per, "start": start}
    url = "https://dataverse.harvard.edu/api/search?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers=dv._build_headers())
    return json.load(urllib.request.urlopen(req, timeout=60))["data"]


existing = set(os.listdir(DEST)) if os.path.isdir(DEST) else set()
seen, pool = set(), []
for st in SUBTREES:
    try:
        total = search_page(st, 0, 1)["total_count"]
    except Exception as e:
        print(f"{st}: size fail {e}", flush=True); continue
    print(f"{st}: {total} datasets", flush=True)
    for start in range(0, total, 100):
        for attempt in range(3):
            try:
                items = search_page(st, start, 100)["items"]; break
            except Exception as e:
                time.sleep(2)
                if attempt == 2:
                    print(f"  {st} page {start} fail: {e}", flush=True); items = []
        for it in items:
            gid = it.get("global_id", "")
            if gid and gid not in seen:
                seen.add(gid)
                dirn = "DVN_" + gid.split("/")[-1]
                if dirn not in existing:
                    pool.append((gid, dirn))
        time.sleep(0.15)
print(f"CENSUS: {len(seen)} unique datasets in the tier, {len(pool)} new to download (rest already on disk)", flush=True)

added = 0
for gid, dirn in pool:
    dest = os.path.join(DEST, dirn)
    try:
        files = dv.list_files(gid)
    except Exception:
        continue
    os.makedirs(dest, exist_ok=True)
    gt = gc = 0
    for f in files:
        df = f.get("dataFile", {})
        fn = df.get("filename", ""); fid = df.get("id"); sz = df.get("filesize", 0) or 0
        low = fn.lower()
        it = low.endswith((".tab", ".csv")); ic = low.endswith((".do", ".r", ".rmd"))
        if not (it or ic) or sz > MAXSZ or fid is None:
            continue
        if it and gt >= 8:
            continue
        if ic and gc >= 8:
            continue
        try:
            dv.fetch_file(fid, dest=os.path.join(dest, os.path.basename(fn)))
            gt += it; gc += ic
        except Exception:
            pass
    added += 1
    if added % 50 == 0:
        print(f"  downloaded {added}/{len(pool)}", flush=True)
    time.sleep(0.12)
print(f"DONE: {added} datasets downloaded; pool now {len(os.listdir(DEST))} dirs", flush=True)
