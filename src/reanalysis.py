#!/usr/bin/env python3
"""Executed re-analysis (paper-2 'value' step): for a confirmed model-data mismatch, refit the
data-appropriate model and show what changes versus the published OLS/linear model. We report,
for the key coefficient, sign and significance under each model, and the share of OLS fitted
values that fall outside the outcome's admissible range (impossible predictions the correct
model cannot make). Coefficient magnitudes differ by scale (OLS is additive; Poisson is on the
log scale; fractional logit on the logit scale), so sign + significance are the comparison.
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
warnings.simplefilter("ignore")


def load(path, sep):
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def keycoef(res, name):
    for p in res.params.index:
        if name in p:
            return res.params[p], res.bse[p], res.pvalues[p]
    return (float("nan"),) * 3


def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."


print("=" * 72)
print("CASE 1  Hate crimes (HRK5HI): count outcome, published as OLS (lm.cluster)")
print("=" * 72)
d = load("data/dv/cnt-ols_hatecrime.csv", ",")[["hcam1", "after", "treat3", "year", "state"]].copy()
for c in ["hcam1", "after", "treat3"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna()
print(f"n={len(d)}  hcam1: mean={d.hcam1.mean():.2f} max={d.hcam1.max():.0f} (integer count, many zeros)")
f = "hcam1 ~ after + after:treat3 + C(year) + C(state)"
ols = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.state})
poi = smf.glm(f, d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": d.state})
for label, res in [("OLS (published)", ols), ("Poisson (correct)", poi)]:
    b, se, p = keycoef(res, "after:treat3")
    print(f"  {label:18}  after:treat3  b={b:+.4f}  se={se:.4f}  p={p:.3f}  {sig(p)}")
neg = (ols.fittedvalues < 0).mean() * 100
print(f"  OLS predicts NEGATIVE hate-crime counts for {neg:.0f}% of observations (impossible); Poisson cannot.")

print()
print("=" * 72)
print("CASE 2  ANC vote share (EKNEIH): proportion [0,1], published as OLS (reg)")
print("=" * 72)
keep = ["anc_vs_na", "share_area_all_tbvc", "share_area_KwaZulu", "year", "cat_b"]
d = pd.read_csv("data/dv/ekneih_full.tab", sep="\t", low_memory=False, usecols=keep)
d.columns = [c.strip().strip('"') for c in d.columns]
for c in ["anc_vs_na", "share_area_all_tbvc", "share_area_KwaZulu"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna()
d = d[(d.anc_vs_na >= 0) & (d.anc_vs_na <= 1)]
print(f"n={len(d)}  anc_vs_na in [{d.anc_vs_na.min():.3f},{d.anc_vs_na.max():.3f}] (bounded vote share)")
f = "anc_vs_na ~ share_area_all_tbvc + share_area_KwaZulu + C(year)"
grp = pd.factorize(d.cat_b)[0]
ols = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": grp})
# fractional logit with the SAME clustered SE as the OLS comparator, so inference is consistent and the
# comparison isolates the model FAMILY (the paper's thesis) rather than confounding it with SE treatment
frac = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": grp})
for label, res in [("OLS (clustered SE)", ols), ("Fractional logit (clustered SE)", frac)]:
    b, se, p = keycoef(res, "share_area_all_tbvc")
    print(f"  {label:28}  share_area_all_tbvc  b={b:+.4f}  se={se:.4f}  p={p:.3f}  {sig(p)}")
oob = ((ols.fittedvalues < 0) | (ols.fittedvalues > 1)).mean() * 100
print(f"  OLS predicts vote shares OUTSIDE [0,1] for {oob:.0f}% of observations (impossible); fractional logit cannot.")

# --- WITHDRAWN false positive (NOT one of Table 4's three re-analyses) --------------------------
# Y0HJJF (African legislative bills) was filed as a count->OLS exemplar but removed from the testbed
# during verification: as the paper reports (Sec. "The verified testbed"), its deposited headline model
# is in fact a correct count model (nbreg), so it is NOT a mismatch. It is kept here only for the
# record and is disabled by default, so this script reproduces exactly the paper's three-row Table 4
# (CASE 1 hate crimes, CASE 2 ANC vote share, CASE 4 battle deaths).
RUN_WITHDRAWN_CASE = False  # set True only to inspect the withdrawn Y0HJJF false positive
if RUN_WITHDRAWN_CASE:
    print()
    print("=" * 72)
    print("CASE 3  [WITHDRAWN] Legislative output (Y0HJJF): filed count->OLS, deposits correct nbreg")
    print("=" * 72)
    d = load("data/dv/cnt-ols_aflegis.tab", "\t")
    clustercol = next((c for c in ["country", "COWcode", "COW"] if c in d.columns), None)
    num = ["bills", "polyarchy_mean", "polyarchy_dev", "laws", "lpop", "wdi_gdpgr", "wdi_lifexp", "year"]
    num = [c for c in num if c in d.columns]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=num + ([clustercol] if clustercol else []))
    print(f"n={len(d)}  bills: mean={d.bills.mean():.1f} max={d.bills.max():.0f} (integer count)  cluster={clustercol}")
    f = "bills ~ polyarchy_mean + polyarchy_dev + laws + lpop + wdi_gdpgr + wdi_lifexp + C(year)"
    g = pd.factorize(d[clustercol])[0]
    ols = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": g})
    poi = smf.glm(f, d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": g})
    for label, res in [("OLS (published)", ols), ("Poisson (correct)", poi)]:
        b, se, p = keycoef(res, "polyarchy_mean")
        print(f"  {label:18}  polyarchy_mean  b={b:+.4f}  se={se:.4f}  p={p:.3f}  {sig(p)}")
    print(f"  OLS predicts NEGATIVE bill counts for {(ols.fittedvalues < 0).mean() * 100:.0f}% of observations.")

print()
print("=" * 72)
print("CASE 4  Civil-war severity (QQRCMD): battle-death COUNTS, published as log-OLS")
print("=" * 72)
d = load("data/dv/cnt-ols_lacina.tab", "\t")
rhs = ["lnduration", "lnpop", "lnmilqual", "lngdp", "cw", "lnmountain", "democ", "ethnicpolar", "relpolar"]
allv = ["battledeadbest", "lnbdb"] + rhs
for c in allv:
    if c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
rhs = [c for c in rhs if c in d.columns]
d = d.dropna(subset=["battledeadbest", "lnbdb"] + rhs)
print(f"n={len(d)}  battle deaths max={d.battledeadbest.max():.0f} (extreme overdispersion)")
ols = smf.ols("lnbdb ~ " + " + ".join(rhs), d).fit()
fnb = "battledeadbest ~ " + " + ".join(rhs)
try:
    nb = smf.negativebinomial(fnb, d).fit(disp=0, maxiter=300); nblab = "NegBin"
except Exception:
    nb = smf.glm(fnb, d, family=sm.families.Poisson()).fit(cov_type="HC1"); nblab = "Poisson/HC1"
for key in ["ethnicpolar", "democ", "relpolar", "lnmilqual"]:
    bo, seo, po = keycoef(ols, key)
    bn, sen, pn = keycoef(nb, key)
    flip = "  <<< SIGNIFICANCE FLIP" if (po < 0.05) != (pn < 0.05) else ""
    print(f"  {key:12}  log-OLS p={po:.3f} {sig(po):4}   {nblab} p={pn:.3f} {sig(pn):4}{flip}")
