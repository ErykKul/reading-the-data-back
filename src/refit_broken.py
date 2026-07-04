#!/usr/bin/env python3
"""Census-wide "provably broken as a model" rate over the clear-tier flags.

For every clear-tier flag (count / proportion outcome under a linear estimator) in atscale_flags.json,
locate the deposited regression command(s) for that (estimator, outcome), parse the covariates, refit
the linear specification on the deposited data (plain OLS; SE options like robust/cluster do not change
fitted values), and measure the share of in-sample fitted values outside the outcome's admissible range
(negative for counts; outside [0,1] for proportions). Generalizes the hate-crime 29%-negative-counts
case (DVN_HRK5HI, used as the calibration anchor) with NO human labeling.

Honesty rules, fixed BEFORE the run:
  - Refit only faithfully reproducible specifications: Stata reg/regress and R lm/lm.cluster/lm_robust/
    aov (+ glm with gaussian/no family), with plain covariate tokens, Stata i.-factors, or R (as.)factor()
    terms. FE/IV/mixed estimators (xtreg, reghdfe, areg, ivreg*, feols, felm, lmer, ...) and commands
    with if/in/weights/subset are SKIPPED and reported as coverage, never guessed at.
  - The refit sample is the data file's complete cases for the used columns (deposited if-conditions and
    pre-filtering are not reconstructed); this approximation is disclosed and checked on the anchor.
  - A flag is judged on ALL its parseable deposited specs (capped): "broken" means at least one deposited
    linear spec on that outcome predicts out-of-range values. Pre-committed primary threshold: >=1% of
    fitted rows out of range ("materially broken"); any-violation and >=5% reported alongside.

Usage: PYTHONPATH=src python src/refit_broken.py [--pool data/dv] [--out atscale_refit_broken.json]
"""
import sys, os, re, glob, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

EPS = 1e-6            # numerical tolerance on range violations
MAX_SPECS = 8         # distinct deposited specs refit per flag, at most
MAX_ROWS = 300_000    # row cap per data file
MAX_LEVELS = 300      # category cap for factor expansion
MIN_N = 30            # minimum complete-case rows

REFIT_STATA = {"reg", "regress"}
REFIT_R = {"lm", "lm.cluster", "lm_robust", "aov", "glm", "glm.cluster"}

STATA_PREFIX = r'(?:xi:\s*|by[^:]*:\s*|bysort[^:]*:\s*|qui(?:etly)?\s+|cap(?:ture)?\s+|noisily\s+|eststo[^:]*:\s*)*'
STATA_CMD = re.compile(r'(?i)^\s*' + STATA_PREFIX + r'\b(reg|regress)\s+(.+)$')
TOKEN = re.compile(r'^[a-zA-Z_]\w*$')
I_FACTOR = re.compile(r'^i(?:b\d+)?\.([a-zA-Z_]\w*)$')          # i.var / ib2.var
R_FACTOR = re.compile(r'^(?:as\.)?factor\(([a-zA-Z_][\w.]*)\)$')
R_CALL = re.compile(r'\b(lm\.cluster|lm_robust|glm\.cluster|aov|lm|glm)\s*\(')


def stata_join_continuations(txt):
    txt = re.sub(r'/\*.*?\*/', ' ', txt, flags=re.S)
    txt = re.sub(r'///.*?\n', ' ', txt)
    return txt


def parse_stata_specs(txt):
    """Yield (est, dep, [covariate terms]) for plain reg/regress lines; None-covariates when unparseable."""
    out = []
    for line in stata_join_continuations(txt).splitlines():
        line = line.split("//")[0]
        m = STATA_CMD.match(line)
        if not m:
            continue
        est, rest = m.group(1).lower(), m.group(2)
        rest = rest.split(",")[0]                                # drop options (vce etc.)
        if re.search(r'\bif\b|\bin\b|\[', rest):                 # if/in/weights -> not reproducible
            out.append((est, None, "condition"))
            continue
        toks = rest.split()
        if not toks:
            continue
        dep, rhs = toks[0].lower(), toks[1:]
        terms = []
        ok = True
        for t in rhs:
            fm = I_FACTOR.match(t)
            if fm:
                terms.append(("cat", fm.group(1).lower()))
            elif TOKEN.match(t):
                terms.append(("num", t.lower()))
            else:                                                # c.x#c.y, L.x, polynomials, wildcards...
                ok = False
                break
        out.append((est, dep, terms if ok else "complex"))
    return out


def balanced_call(txt, start):
    """Return the argument string of the call whose '(' is at txt[start], paren-balanced, capped."""
    depth, i = 0, start
    for i in range(start, min(len(txt), start + 2000)):
        if txt[i] == "(":
            depth += 1
        elif txt[i] == ")":
            depth -= 1
            if depth == 0:
                return txt[start + 1:i]
    return None


def parse_r_specs(txt):
    out = []
    for m in R_CALL.finditer(txt):
        fn = m.group(1).lower()
        args = balanced_call(txt, m.end() - 1)
        if args is None:
            continue
        if fn.startswith("glm"):
            fam = re.search(r'family\s*=\s*[\'"]?(\w+)', args)
            if fam and fam.group(1).lower() != "gaussian":
                continue                                         # correctly specified glm, not our flag
        if re.search(r'\bsubset\s*=', args):
            out.append((fn, None, "condition"))
            continue
        fm = re.match(r'\s*(?:formula\s*=\s*)?([a-zA-Z_][\w.]*)\s*~\s*([^,]*)', args)
        if not fm:
            continue
        dep, rhs = fm.group(1).lower(), fm.group(2).strip()
        if rhs in ("1", "."):
            out.append((fn, dep, "complex"))
            continue
        terms, ok = [], True
        for t in [x.strip() for x in re.split(r'\+', rhs)]:
            rf = R_FACTOR.match(t)
            im = re.match(r'^([a-zA-Z_][\w.]*)\s*:\s*([a-zA-Z_][\w.]*)$', t)
            if rf:
                terms.append(("cat", rf.group(1).lower()))
            elif im:                                             # plain numeric a:b product
                terms.append(("int", im.group(1).lower(), im.group(2).lower()))
            elif re.match(r'^[a-zA-Z_][\w.]*$', t):
                terms.append(("num", t.lower()))
            else:                                                # I(), log(), splines, * expansions...
                ok = False
                break
        out.append((fn, dep, terms if ok else "complex"))
    return out


def dataset_specs(ddir):
    specs = []
    for f in glob.glob(os.path.join(ddir, "**", "*"), recursive=True):
        low = f.lower()
        try:
            if low.endswith(".do"):
                specs += parse_stata_specs(open(f, errors="ignore").read())
            elif low.endswith((".r", ".rmd")):
                specs += parse_r_specs(open(f, errors="ignore").read())
        except Exception:
            pass
    return specs


def load_table(path):
    sep = "\t" if path.lower().endswith(".tab") else ","
    df = pd.read_csv(path, sep=sep, nrows=MAX_ROWS, low_memory=False,
                     encoding_errors="replace", on_bad_lines="skip")
    if df.shape[1] == 1 and sep == ",":                          # semicolon csv fallback
        df = pd.read_csv(path, sep=";", nrows=MAX_ROWS, low_memory=False,
                         encoding_errors="replace", on_bad_lines="skip")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_file(ddir, needed):
    """Largest tabular file whose header contains all needed columns (case-insensitive)."""
    cands = []
    for f in glob.glob(os.path.join(ddir, "**", "*"), recursive=True):
        if not f.lower().endswith((".tab", ".csv")) or os.path.getsize(f) == 0:
            continue
        try:
            sep = "\t" if f.lower().endswith(".tab") else ","
            head = pd.read_csv(f, sep=sep, nrows=0, encoding_errors="replace")
            cols = {str(c).strip().lower() for c in head.columns}
            if len(cols) <= 1 and sep == ",":
                head = pd.read_csv(f, sep=";", nrows=0, encoding_errors="replace")
                cols = {str(c).strip().lower() for c in head.columns}
            if needed <= cols:
                cands.append((os.path.getsize(f), f))
        except Exception:
            pass
    return max(cands)[1] if cands else None


def term_vars(terms):
    out = set()
    for t in terms:
        out.update(t[1:])
    return out


def design_matrix(df, terms):
    """Complete-case design matrix for the parsed terms; None + reason when not buildable."""
    parts = []
    for t in terms:
        kind, v = t[0], t[1]
        if kind == "int":                                        # numeric a:b product
            a = pd.to_numeric(df[t[1]], errors="coerce")
            b = pd.to_numeric(df[t[2]], errors="coerce")
            if a.notna().mean() < 0.5 or b.notna().mean() < 0.5:
                return None, None, "interaction_nonnumeric"
            parts.append((a * b).rename(f"{t[1]}:{t[2]}").to_frame())
            continue
        col = df[v]
        num = pd.to_numeric(col, errors="coerce")
        is_num = num.notna().mean() >= 0.5
        if kind == "num" and is_num:
            parts.append(num.rename(v).to_frame())
        else:                                                    # categorical (declared, or string column)
            if col.nunique(dropna=True) > MAX_LEVELS:
                return None, None, "too_many_levels"
            d = pd.get_dummies(col.astype("string"), prefix=v, drop_first=True, dtype=float)
            d[col.isna()] = np.nan
            parts.append(d)
    X = pd.concat(parts, axis=1)
    return X, X.notna().all(axis=1), None


def refit_one(df, outcome, terms, kind):
    y = pd.to_numeric(df[outcome], errors="coerce")
    X, rows_ok, err = design_matrix(df, terms)
    if err:
        return {"status": err}
    keep = rows_ok & y.notna()
    n = int(keep.sum())
    Xk, yk = X[keep].to_numpy(float), y[keep].to_numpy(float)
    p = Xk.shape[1] + 1
    if n < MIN_N or n < 3 * p:
        return {"status": "too_few_rows", "n": n, "p": p}
    # the violation claim is only airtight if the outcome itself never leaves its admissible domain
    # in the refit sample (sentinel codes past the profiler's row cap etc. would void "impossible")
    if kind == "count" and (yk.min() < -EPS):
        return {"status": "outcome_domain_mismatch", "n": n}
    if kind == "proportion" and (yk.min() < -EPS or yk.max() > 1 + EPS):
        return {"status": "outcome_domain_mismatch", "n": n}
    A = np.column_stack([np.ones(n), Xk])
    beta, *_ = np.linalg.lstsq(A, yk, rcond=None)
    fit = A @ beta
    if kind == "count":
        viol = fit < -EPS
    else:                                                        # proportion
        viol = (fit < -EPS) | (fit > 1 + EPS)
    return {"status": "refit_ok", "n": n, "p": p, "viol_share": float(viol.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/dv")
    ap.add_argument("--flags", default="atscale_flags.json")
    ap.add_argument("--out", default="atscale_refit_broken.json")
    ap.add_argument("--only", default=None, help="comma-separated dataset dirs (debug)")
    args = ap.parse_args()

    res = json.load(open(args.flags))
    todo = []                                                    # (dataset, flag) for every clear-tier flag
    for d, x in res.items():
        for f in x["flags"]:
            if f["severity"] == "clear":
                todo.append((d, f))
    if args.only:
        keep = set(args.only.split(","))
        todo = [t for t in todo if t[0] in keep]
    print(f"clear-tier flags to assess: {len(todo)} across {len({d for d, _ in todo})} datasets", flush=True)

    out, spec_cache, df_cache = [], {}, {}
    for i, (d, f) in enumerate(todo):
        ddir = os.path.join(args.pool, d)
        rec = {"dataset": d, "estimator": f["estimator"], "outcome": f["outcome"], "kind": f["kind"]}
        est, dep = f["estimator"].lower(), f["outcome"].lower()
        base = est.split("(")[0]
        if base not in REFIT_STATA | REFIT_R:
            rec["status"] = "estimator_not_refittable"           # FE / IV / mixed etc., by design
            out.append(rec)
            continue
        if d not in spec_cache:
            spec_cache[d] = dataset_specs(ddir)
        matches = [s for s in spec_cache[d] if s[0] == base and s[1] == dep]
        if not matches:
            rec["status"] = "no_code_line"
            out.append(rec)
            continue
        parseable = [t for _, _, t in matches if isinstance(t, list)][:MAX_SPECS]
        if not parseable:
            reasons = {t for _, _, t in matches if isinstance(t, str)}
            rec["status"] = "condition" if "condition" in reasons else "complex_rhs"
            out.append(rec)
            continue
        results = []
        for terms in parseable:
            needed = {dep} | term_vars(terms)
            fkey = (d, tuple(sorted(needed)))
            try:
                if fkey not in df_cache:
                    path = find_file(ddir, needed)
                    df_cache[fkey] = load_table(path) if path else None
                df = df_cache[fkey]
                if df is None:
                    results.append({"status": "covariate_missing"})
                    continue
                results.append(refit_one(df, dep, terms, f["kind"]))
            except Exception as e:
                results.append({"status": "load_fail", "err": str(e)[:80]})
        oks = [r for r in results if r["status"] == "refit_ok"]
        if oks:
            best = max(oks, key=lambda r: r["viol_share"])
            rec.update(status="refit_ok", n_specs=len(oks),
                       viol_share=best["viol_share"], n=best["n"], p=best["p"])
        else:
            rec["status"] = results[0]["status"] if results else "no_code_line"
        out.append(rec)
        if len(df_cache) > 40:                                   # keep memory bounded
            df_cache.clear()
        if i % 50 == 0:
            done = sum(1 for r in out if r["status"] == "refit_ok")
            print(f"  {i}/{len(todo)} assessed, {done} refit_ok", flush=True)

    # pre-committed summary (recomputable from the per-flag records by reproduce.py)
    okr = [r for r in out if r["status"] == "refit_ok"]
    summ = {"clear_flags": len(out),
            "refit_ok": len(okr),
            "status_taxonomy": {},
            "broken_any": sum(1 for r in okr if r["viol_share"] > 0),
            "broken_ge1pct": sum(1 for r in okr if r["viol_share"] >= 0.01),
            "broken_ge5pct": sum(1 for r in okr if r["viol_share"] >= 0.05)}
    for r in out:
        summ["status_taxonomy"][r["status"]] = summ["status_taxonomy"].get(r["status"], 0) + 1
    for kind in ("count", "proportion"):
        k = [r for r in okr if r["kind"] == kind]
        summ[kind] = {"refit_ok": len(k),
                      "broken_ge1pct": sum(1 for r in k if r["viol_share"] >= 0.01)}
    json.dump({"summary": summ, "flags": out}, open(args.out, "w"), indent=0)
    print(json.dumps(summ, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
