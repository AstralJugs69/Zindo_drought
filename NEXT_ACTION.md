# Next action — corrected validation results

- Branch: `codex/validation-rebuild`. Kaggle's current clean checkout is pinned
  to `8c6edafb3e829b8d71f4b3cc31d249ad4e2d2aae` for the completed R04 run.
- Notebook: a running Draft Session with safe idempotent bootstrap, checked
  dataset discovery, one thin named runner, and one concise artifact cell. The
  old submission-training cell was never run. Run All and Save Version were not
  used.
- Local checks passed: Python syntax checks and eight simulator/metric tests,
  including calendar gaps, multi-cycle masks, target-free label alignment,
  streamed-history missing values, and the real Test's 280,961 IDs/h counts.
- Completed Kaggle artifacts (all session-local; download links were clicked but
  no matching local file appeared, so they are not claimed locally preserved):
  - `preflight_20260907T143613Z`: repaired R00 persistence package SHA-256
    `8d37c97dd6d5094ceb15410b8976e0004032a2261fa7f57f26d9a4abca21a867`.
  - `r01_20260907T145404Z`: primary development recipe, package SHA-256
    `010f0776315732a387924aa7010dc666c04056faed15196338344eb44f28c4ec`.
  - `r02_20260907T150052Z`: rejected schedule-training test, SHA-256
    `904a28a06476429a2d8c70a265eee3161ea4a6d276bff021c740540b8fd717ba`.
  - `r03_20260907T151039Z`: rejected visible-history test, SHA-256
    `f6acc20203c8be7944c24c19bc4f2077b8d5b3760ed69f22671cd3c32beabfda`.
  - `r04_20260907T151713Z`: rejected/mixed absolute-target test, SHA-256
    `f9ecd869e4e9e888252ac74ce83660932a6aba96b3a01f9b83488ea578b0a8b6`.
- Weighted RMSE by replay (Jan-2009 / Sep-2006 / Sep-2007): R00 persistence
  `0.711328 / 0.638469 / 0.625816`; R01 `0.573954 / 0.545595 / 0.537405`;
  R02 `0.613719 / 0.583597 / 0.573021`; R03 `0.616904 / 0.600467 / 0.576177`;
  R04 `0.573525 / 0.544811 / 0.539309`.
- Decision: retain R01 (safe delta target, legacy deterministic h sampler,
  fixed 173 rounds) as the primary **development** recipe; persistence is the
  transparent fallback. Do not promote R02/R03/R04. The Dec-2014--Jun-2015
  block has real h=1..13 and is infeasible as an h=1..7 confirmation fold.
- No Test prediction CSV or Zindi submission is authorized/generated. The
  historical public EXP005/EXP010 scores remain records only.
- If continuing, first create a newly predeclared independent confirmation
  scenario with full h=1..7 support or explicitly accept the absence of an
  outer confirmation; do not tune R01 further against the current three
  development replays.
