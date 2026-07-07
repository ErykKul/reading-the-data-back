#!/usr/bin/env python3
"""P2 detection-through-noise pipeline. For each Dataverse dataset dir:
  (1) profile every tabular file -> {variable: distributionKind} via the extended DDI-CDI classifier (measured side),
  (2) extract (estimator, outcome) pairs from deposited .do/.R code (assumed side, testbed-verified keyword route),
  (3) flag a mismatch when a LINEAR estimator is applied to a non-continuous outcome (count/proportion/binary/ordinal).
Runs blind over a whole pool; measures recovery of the embedded 40-case testbed + flag volume.

Usage: PYTHONPATH=src python src/atscale_pipeline.py <pool_dir> [--out flags.json]
"""
import sys, os, re, glob, json, argparse
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from cdi_generator_ext import stream_profile_csv

# assumed distribution by estimator family
LINEAR = {"reg", "regress", "reghdfe", "xtreg", "xtscc", "areg", "ivreg2", "ivreg", "ivregress", "newey",
          "reg2hdfe", "xtivreg", "lm", "feols", "felm", "lmer", "rlm", "lm.cluster",
          "aov", "anova", "lm_robust", "lm_lin", "ols", "sureg", "mixed", "xtmixed", "hetregress"}  # assume continuous/Gaussian
COUNT_EST = {"poisson", "nbreg", "fepois", "xtpoisson", "xtnbreg", "ppml", "ppmlhdfe", "glm.nb"}  # ok for count
BINARY_EST = {"logit", "probit", "xtlogit", "clogit", "logistic", "biprobit", "cloglog", "glmer"}  # ok for binary
PROP_EST = {"betareg", "fracreg", "dtobit", "freg"}                                             # ok for proportion
ORD_EST = {"ologit", "oprobit", "gologit", "gologit2", "polr"}                                  # ok for ordinal
NONCONT = {"count", "proportion", "binary", "ordinal"}

STATA_RE = re.compile(
    r'(?im)^\s*(?:xi:\s*|by[^:]*:\s*|qui(?:etly)?\s+|cap(?:ture)?\s+|eststo[^:]*:\s*)*'
    r'\b(reghdfe|regress|xtreg|xtscc|areg|ivreg2|ivregress|ivreg|newey|reg|sureg|mixed|xtmixed|hetregress|'
    r'poisson|nbreg|fepois|xtpoisson|'
    r'xtnbreg|ppmlhdfe|ppml|logit|probit|ologit|oprobit|mlogit|clogit|betareg|fracreg|logistic)\b'
    r'\s+\(?\s*([a-zA-Z_]\w*)')
R_RE = re.compile(r'(?is)\b(lm\.cluster|glm\.cluster|lm_robust|lm_lin|feols|felm|glmer|lmer|betareg|polr|aov|anova|lm|glm)\s*\(([^)]{0,300})')
R_FORMULA = re.compile(r'([a-zA-Z_][\w.]*)\s*~')


def profile_dataset(ddir, cap=120000):
    kinds = {}
    for f in glob.glob(os.path.join(ddir, "**", "*"), recursive=True):
        if f.lower().endswith((".tab", ".csv")) and os.path.getsize(f) > 0:
            try:
                cols, stats, *_ = stream_profile_csv(Path(f), header="auto", limit_rows=cap, compute_md5=False)
                for name, st in zip(cols, stats):
                    k = st.distribution_kind()
                    nm = name.strip().lower()
                    if nm and (nm not in kinds or kinds[nm] == "categorical"):
                        kinds[nm] = k
            except Exception:
                pass
    return kinds


def r_family_is_linear(call_args):
    fam = re.search(r'family\s*=\s*[\'"]?(\w+)', call_args)
    if fam:
        return fam.group(1).lower() in ("gaussian",)   # glm(family=gaussian) is linear; poisson/binomial are not
    return True  # glm/lm with no family -> linear (lm always linear)


def extract_regressions(ddir):
    regs = []
    for f in glob.glob(os.path.join(ddir, "**", "*"), recursive=True):
        if not f.lower().endswith((".do", ".r", ".rmd")):
            continue
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        for m in STATA_RE.finditer(txt):
            regs.append((m.group(1).lower(), m.group(2).lower()))
        for m in R_RE.finditer(txt):
            fn, args = m.group(1).lower(), m.group(2)
            fm = R_FORMULA.search(args)
            if not fm:
                continue
            dep = fm.group(1).lower()
            if fn in ("glm", "glm.cluster") and not r_family_is_linear(args):
                fn = "glm.nonlinear"   # correctly specified glm, not a linear-on-noncont flag
            regs.append((fn, dep))
    return regs


def assumed_kind(est):
    base = est.split(".")[0].split("(")[0].split("-")[0]  # also strip annotations like "regress-logOLS"
    if est in LINEAR or base in LINEAR:
        return "continuous"
    if base in COUNT_EST or est in COUNT_EST:
        return "count"
    if base in BINARY_EST or est in BINARY_EST:
        return "binary"
    if base in PROP_EST:
        return "proportion"
    if base in ORD_EST:
        return "ordinal"
    return None  # unknown / correctly-specified nonlinear (e.g. glm.nonlinear)


def detect(ddir):
    kinds = profile_dataset(ddir)
    regs = extract_regressions(ddir)
    flags, matched = [], 0
    seen = set()
    for est, dep in regs:
        if assumed_kind(est) != "continuous":
            continue
        # transformed outcomes (log/ln) we cannot profile directly are counted as outcome-id misses
        mk = kinds.get(dep) or kinds.get(dep.lstrip("l_").lstrip("ln"))
        if mk is None:
            continue
        matched += 1
        if mk in NONCONT and (est, dep, mk) not in seen:
            seen.add((est, dep, mk))
            flags.append({"estimator": est, "outcome": dep, "kind": mk,
                          "severity": "clear" if mk in ("count", "proportion") else "debated"})
    return {"kinds_n": len(kinds), "regs_n": len(regs), "matched_outcomes": matched, "flags": flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--out", default="atscale_flags.json")
    args = ap.parse_args()
    dirs = sorted([d for d in glob.glob(os.path.join(args.pool, "*")) if os.path.isdir(d)])
    results = {}
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        results[name] = detect(d)
        if i % 20 == 0:
            print(f"  {i}/{len(dirs)} {name}: {len(results[name]['flags'])} flags", flush=True)
    json.dump(results, open(args.out, "w"), indent=0)
    tot_flags = sum(len(r["flags"]) for r in results.values())
    flagged_ds = sum(1 for r in results.values() if r["flags"])
    print(f"\nPOOL: {len(dirs)} datasets | {flagged_ds} flagged | {tot_flags} total flags")


if __name__ == "__main__":
    main()
