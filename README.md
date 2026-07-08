# Reading the Data Back: Detecting Model-Data Misspecification at Repository Scale (reproduction package)

Applied quantitative research keeps fitting linear, Gaussian models to outcomes that are not
continuous: event counts, bounded proportions, binary and ordinal responses. The detection rule is old
and fits in one line (flag any linear estimator applied to a non-continuous outcome); what this package
reproduces is that rule **operationalized over standard repository metadata** and measured on a real
archive. A production DDI-CDI generator is extended so every variable carries a measured
`distributionKind` (the resolution the standard's datatype vocabulary cannot express); the assumed side
is read from the deposited Stata/R analysis code; a second, objective stage refits flagged models and
checks whether they predict impossible values on their own data.

**Reproduction runs offline: no API key, no network.** Every artifact `reproduce.py` reads (census flag
records, the verified testbed CSV, the full two-rater adjudication record, the refit results) is
committed in this repository. The raw ~20 GB census pool is not redistributed; it rebuilds from public
DOIs with a shipped script (see below).

## The headline (census of the political-science replication tier)

| measurement | result |
|---|---|
| corpus (all six journal collections: AJPS, JOP, APSR, PA, ISQ, CPS) | 4,882 datasets, 567,860 profiled variables |
| recovery through noise (dataset-level, blind) | 14 of 21 in-census verified mismatch datasets (67%; 14 of 24 overall) |
| controls | all 6 in-census verified-correct datasets pass |
| flag volume | 965 datasets flagged (198 per 1,000 raw; 67% of the flaggable base); 4,233 flags, 944 clear-tier |
| precision on novel clear-tier flags (two-rater, blind, kappa 0.90) | 55% overall; 62% proportions, 49% counts |
| provably broken as a model (automated OLS refit, no labeling) | 54 of 244 faithfully refittable clear-tier flags predict out-of-range values on >=1% of their own rows |

## Quickstart

Python 3.12+ (developed and verified on 3.14; `.tool-versions` pins the verified interpreter for
mise/asdf users, but any recent CPython works).

```
make setup        # one time: create .venv and install requirements
make reproduce    # every paper number, from the committed artifacts
```

or without make:

```
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` prints every figure the paper cites under a banner naming the paper item. The two
sections that need the raw census pool (outcome-ID accuracy; the hate-crimes re-analysis anchor) are
marked SKIPPED unless `data/dv/` is present, and recompute automatically when it is.

## Where each paper number comes from

`reproduce.py` is the single driver and the source of truth the prose must match. The scripts it
summarizes, each runnable alone from the package root (`PYTHONPATH=src python src/<script>.py`):

| paper item | script |
|---|---|
| the enriched DDI-CDI generator: per-variable `distributionKind` in one streaming pass | `cdi_generator_ext.py` (fork of `cdi_generator_jsonld.py` from [libis/rdm-integration](https://github.com/libis/rdm-integration)) |
| standalone profiler used to build and re-verify the testbed | `profile_outcome.py` |
| testbed precision/recall and the disclosed one-line-baseline tie | `run_eval_v2.py` (+ `testbed_eval.py`) |
| census harvest (all 4,882 datasets, resumable) | `census_harvest.py` (Dataverse API client: `dv.py`) |
| census profiling + flagging -> `atscale_flags.json` | `atscale_pipeline.py` |
| recovery, denominators, flag volume, novel-flag extraction | `atscale_measure.py` |
| the balanced adjudication sample + the two-rater adjudication panel | `prep_census_adjudication.py`, `census_adjudication_panel.js` |
| executed re-analysis of verified cases: sign/significance under the correct model | `reanalysis.py` |
| automated refit check: out-of-range fitted values, coverage taxonomy | `refit_broken.py` |

Committed artifacts read by `reproduce.py`:

- `atscale_flags.json` - the full census flag record (4,882 datasets)
- `datasets/operator2_testbed_v2.csv` - the verified 40-case testbed: DOIs, outcome variables,
  recorded classes, estimators, severity tiers, evidence notes, all re-verified against the live archive
- `atscale_census_verdicts.json` - the complete two-rater adjudication record (all 160 verdicts,
  each with its evidence note and false-positive class), so every call can be re-examined
- `atscale_refit_broken.json` - the automated refit results with the full coverage taxonomy

The remaining `atscale_*.json` files are the census-round intermediates: `atscale_novel_flags.json`
(the novel clear-tier flags `atscale_measure.py` extracts) and `atscale_census_sample.json` /
`atscale_census_irr.json` (the balanced sample + inter-rater overlap that `prep_census_adjudication.py`
builds for the panel). `reproduce.py` recomputes the honest kappa from `atscale_census_verdicts.json`
and shows the favorable collapse it replaces.

## What is in the package

```
reading-the-data-back/
  reproduce.py                  one command -> every paper number
  datasets/
    operator2_testbed_v2.csv      the verified testbed (40 cases, 30 packages, DOIs + evidence)
  atscale_flags.json            census flag record        \
  atscale_census_verdicts.json  full adjudication record   | the committed artifacts
  atscale_refit_broken.json     automated refit results    |
  atscale_*.json                census-round intermediates /
  src/                          pipeline, generator extension, evaluation + fetch scripts
  requirements.txt  Makefile  example.env  SHA256SUMS
```

## What is bundled, and redistribution

This package ships **derived artifacts and our own labels only**: flag records, adjudication verdicts,
refit summaries, and the testbed CSV (our severity labels over public DOIs). **No deposited research
data or analysis code is redistributed.** The census pool rebuilds locally from the public Harvard
Dataverse DOIs:

```
make corpus       # ~20 GB into data/dv/, resumable; API key optional (example.env)
```

Every dataset DOI is in `atscale_flags.json` and the testbed CSV, so single datasets can also be
fetched and checked individually (`src/dv.py`).

## Integrity

```
sha256sum -c SHA256SUMS
```

Expected output of `make reproduce`: each stage prints its numbers to stdout under a banner naming the
paper item, ending with the refit summary self-check (`stored summary matches recomputation: OK`).

## Status, archive and citation

This repository is the public record of this work while the accompanying manuscript is under review:
the full method, every reported number, and everything needed to reproduce them offline. A frozen copy
will be deposited in the KU Leuven Research Data Repository (RDR) with a DOI at publication; until then,
cite this repository directly (URL and commit). The full paper citation will be added here at publication.

## License

Code: Apache-2.0. Derived data (flag records, adjudication verdicts, testbed labels): CC-BY-4.0.
The deposited datasets and analysis code the pipeline audits keep their original licenses and are
never redistributed here; they are fetched from their public DOIs.
