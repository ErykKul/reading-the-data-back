export const meta = {
  name: 'census-adjudication-two-rater',
  description: 'Census-scale two-rater adjudication of clear-tier flags (precision + inter-rater kappa)',
  phases: [{ title: 'RaterA', detail: 'primary rater over the full stratified sample' }, { title: 'RaterB', detail: 'independent second rater over the IRR subsample' }],
}
const P = '.'  // the package root: atscale_census_sample.json alongside, the rebuilt pool in data/dv
const POOL = `${P}/data/dv`
const SCHEMA = {
  type: 'object',
  properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
    dataset: { type: 'string' }, outcome: { type: 'string' }, kind: { type: 'string' },
    real: { type: 'boolean' }, fp_class: { type: 'string' }, reason: { type: 'string' } },
    required: ['dataset', 'outcome', 'real', 'fp_class', 'reason'] } } },
  required: ['verdicts'],
}
// shared rubric: the refined classifier already routes single-item rescaled Likerts to ordinal, so clear-tier
// proportion flags should be native shares/rates; raters still decide each case on merits.
const RUBRIC = `A flag {dataset, estimator, outcome, kind} is REAL iff, verified in the deposited code and data: the linear estimator (reg/regress/xtreg/lm/felm/lm_robust/aov) is applied to <outcome> as the DEPENDENT variable, AND the outcome is genuinely either a bounded PROPORTION/rate/share in [0,1] (Papke-Wooldridge; strongest with mass at 0 or 1; a genuine share of a bounded total or a rate, NOT a single Likert item rescaled to [0,1]) OR a genuine EVENT COUNT (King 1988; a non-negative integer tally, Poisson-like, not a large near-continuous magnitude and not an integer covariate like age/income). FALSE classes: not_outcome (identifier/predictor, not the DV); not_count (continuous/covariate integer, or large near-continuous count where OLS is standard); not_proportion (not truly bounded, or a share where OLS is defensible); rescaled_ordinal (a Likert/rating item min-max-rescaled to [0,1], where interval OLS is a defensible mainstream choice); interior_defensible (a [0,1] variable strictly interior with no boundary mass); transformed/correct_model (a log transform or the correct family was actually used); cannot_verify (files inaccessible). Decide the rescaled-ordinal case on its merits; do not assume [0,1] implies real.`
const promptFor = (file, s, e, who) => `You are ${who}, adjudicating model-data-misspecification flags against deposited data and code. Apply this rubric strictly and consistently:\n\n${RUBRIC}\n\nRead the flag file: \`cat ${file}\` (a JSON array). Adjudicate ONLY indices ${s} through ${e - 1} inclusive. For each: inspect \`${POOL}/<dataset>/\` code + data, sample the outcome's actual values, and return one verdict with a one-sentence grounded reason.`

function ranges(n, b) { const r = []; for (let i = 0; i < n; i += b) r.push([i, Math.min(i + b, n)]); return r }

// Rater A over the full stratified sample. args.nA = sample size (passed in).
phase('RaterA')
// sample sizes: prep wrote 120 (60 count + 60 proportion) and a 40-flag IRR subsample. args fallback for reuse.
const NA = (args && args.nA) || 120, NB = (args && args.nB) || 40, B = 5
log(`adjudicating: rater A over ${NA}, rater B (IRR) over ${NB}`)
const aRes = await parallel(ranges(NA, B).map(([s, e]) => () =>
  agent(promptFor(`${P}/atscale_census_sample.json`, s, e, 'the primary rater (Rater A)'),
    { label: `A:${s}-${e}`, phase: 'RaterA', schema: SCHEMA })))
const aVerdicts = aRes.filter(Boolean).flatMap(r => r.verdicts || [])

// Rater B, independent, over the IRR subsample only.
phase('RaterB')
const bRes = await parallel(ranges(NB, B).map(([s, e]) => () =>
  agent(promptFor(`${P}/atscale_census_irr.json`, s, e, 'an INDEPENDENT second rater (Rater B) who has not seen any other judgments'),
    { label: `B:${s}-${e}`, phase: 'RaterB', schema: SCHEMA })))
const bVerdicts = bRes.filter(Boolean).flatMap(r => r.verdicts || [])

const prec = (vs) => ({ n: vs.length, real: vs.filter(v => v.real).length, p: vs.length ? vs.filter(v => v.real).length / vs.length : 0 })
const byKind = {}
for (const v of aVerdicts) { const k = v.kind || '?'; byKind[k] = byKind[k] || { n: 0, real: 0 }; byKind[k].n++; if (v.real) byKind[k].real++ }
return { raterA: prec(aVerdicts), byKind, raterB: prec(bVerdicts), aVerdicts, bVerdicts }
