# CLAUDE.md — AdaptiveCX Stage 2: Customer State Model

Design doc for Stage 2, written against `AdaptiveCX_ML_CX_Flow_Review.docx` (the
team's proposed 2-stage architecture). No separate Stage-2 spec exists yet, so
this file *is* that spec, kept in its own project so it doesn't mix with the
already-complete, already-validated Stage 1 (`adaptivecx-stage1/`).

## 0. What Stage 2 is (per the review doc, §6-§9)

> Goal: use Stage-1 emotional outputs plus acoustic/interaction features to
> estimate customer-experience states.
> Stage 2 model: combined feature vector → XGBoost / LightGBM → Stress,
> frustration, urgency, escalation risk.

Stage 2 does **not** reprocess raw audio through a neural encoder. It's a
tabular gradient-boosting model on top of:

1. Stage 1's outputs (emotion probabilities, arousal, valence) — reused from
   the already-trained `adaptivecx-stage1/models/best_stage1.pt`, **not
   retrained**.
2. Acoustic features computed directly from the waveform (pitch, energy,
   speaking rate, pause/silence structure) — no separate model needed for
   these, per review doc §5.

This also matches the real-time constraint: XGBoost inference is
sub-millisecond, unlike re-running a neural sequence model per turn.

## 1. The label problem (review doc §7-§8, honestly addressed)

No public dataset pairs real customer-service voice with real stress /
frustration / urgency / escalation-risk labels:

- **EmoWork** (Kaist-ICLab, *Scientific Data* 2025) is a real, well-matched
  dataset — call-center workers reacting to actors playing dissatisfied
  customers, with genuine self-reported stress/arousal/valence — but it's
  gated behind a Data Use Agreement + application process on Zenodo
  (CC BY-NC-ND license, 60GB). **Not usable on this project's timeline.**
  Worth the team formally requesting access in parallel, as a future
  upgrade path — see §6 below for exactly what would change if/when access
  is granted.
- **"V2V"** (named in the review doc) could not be identified as a specific
  accessible public dataset via search. Confirm the exact source with the
  trainer before assuming it's usable.
- IEMOCAP (Stage 1's dataset) has real emotion/arousal/valence/dominance
  labels, but no stress/frustration/urgency/escalation-risk annotations at
  all — those concepts don't exist in its label set.

**Per the review doc's own explicit fallback (§7, §84): rule-generated
labels are allowed for the prototype, as long as they are documented as a
bootstrap/prototype scoring layer and never presented as ground truth.**
That's what this does.

## 2. Bootstrap CX-label formulas (documented, not hidden)

Computed per IEMOCAP utterance from Stage-1 outputs + acoustic features
(all sub-terms normalized to `[0, 1]` first):

```
neg_emotion   = P(angry) + P(sad)                       # Stage 1 output
high_arousal  = (arousal + 1) / 2                        # Stage 1 output, [-1,1] -> [0,1]
neg_valence   = (1 - valence) / 2                         # Stage 1 output, [-1,1] -> [0,1]
pitch_var     = normalized F0 std-dev across the utterance
speak_rate    = normalized syllable-nuclei rate (energy-envelope peaks / sec)
pause_ratio   = silence duration / total duration
energy_level  = normalized mean RMS energy

stress          = 0.35*neg_emotion + 0.30*high_arousal + 0.20*pitch_var + 0.15*(1 - pause_ratio)
frustration     = 0.40*P(angry)     + 0.25*high_arousal + 0.20*neg_valence + 0.15*speak_rate
urgency         = 0.40*speak_rate   + 0.30*high_arousal + 0.20*(1 - pause_ratio) + 0.10*energy_level
escalation_risk = 0.5*stress + 0.5*frustration
```

Rationale for each weight is the same kind of hand-designed, documented
heuristic already used in `agent/policy_engine.py` / `agent/emotion_engine.py`
for the live agent's *text*-based signals — this is the acoustic analogue.
Weights are a starting point, tunable, and printed in the notebook so
they're never silently baked in.

**What XGBoost is actually learning:** a smooth, generalizable approximation
of this formula from the fused feature vector — not real customer stress.
Evaluation metrics (MAE/RMSE/R²) measure fit quality to the bootstrap
formula, not real-world accuracy. This must be stated plainly in the demo,
matching review-doc item #14 ("this is a proposed plan, not a claim that
every label is real ground truth").

## 2b. Text-model-judged labels (Section 9b in the notebook — an upgrade path, added after the first run)

The formula in §2 is a fixed linear combination — transparent, but crude,
and acoustic-only (it never looks at *what was said*, only how). The
notebook also supports a second, complementary bootstrap source: a
pretrained fine-grained text-emotion classifier
(`SamLowe/roberta-base-go_emotions`, 28 GoEmotions categories, via
`transformers`) reads each utterance's real IEMOCAP transcript, and a
documented mapping combines its emotion probabilities into
stress/frustration/urgency/escalation_risk.

**Originally designed as a Claude API call** (per-transcript LLM judgment),
but switched to this free local classifier because the user did not have
separate Anthropic API billing set up (a Claude.ai subscription does not
include API credits — they're billed separately) and didn't want to spend
Claude Code usage doing ~1200 judgments manually in-conversation either.
This local model needs no API key, no billing, no Kaggle Secret — runs
on-device (GPU if available) same as everything else in the notebook.

**This is still not real ground truth** — it replaces one heuristic (the
acoustic formula) with a second, independent one (a pretrained classifier's
text-emotion probabilities, mapped by a documented formula), not measured
human ratings of real customer calls. Documented and labeled as a
bootstrap/prototype layer exactly like §2, per the review doc's own rule
(§84).

- **Label mapping** (GoEmotions clusters -> CX targets, all sub-terms
  clipped to `[0,1]`):
  ```
  anger_cluster       = anger + annoyance
  fear_cluster        = fear + nervousness
  sad_cluster         = sadness + disappointment + grief + remorse
  disapproval_cluster = disgust + disapproval

  stress          = 0.35*fear_cluster + 0.30*anger_cluster + 0.20*sad_cluster + 0.15*surprise
  frustration     = 0.45*anger_cluster + 0.25*disapproval_cluster + 0.20*sad_cluster + 0.10*confusion
  urgency         = 0.40*fear_cluster + 0.30*anger_cluster + 0.20*surprise + 0.10*(1 - neutral)
  escalation_risk = 0.5*stress + 0.5*frustration
  ```
- **Coverage**: every utterance with a matching transcript in each split
  (not a small sample — free/local, so no cost reason to subsample).
- **Domain mismatch caveat carries over**: IEMOCAP is acted general
  dialogue, not real customer-service audio.
- **Output**: a second set of models (`models/{target}_textmodel.json`), a
  second evaluation report (`results/evaluation_textmodel.json`), and a
  Pearson correlation between the acoustic-formula and text-model labels on
  the same utterances — a useful diagnostic for whether the two
  independent heuristics (audio prosody vs. spoken content) roughly agree
  or diverge. Divergence is expected and informative, not a bug.
- `CXStateModel(cx_models=text_cx_models, feature_cols=FEATURE_COLS)` swaps
  in the text-model-labeled variant in place of the formula-labeled
  default.

## 3. Features (review doc §5)

| Feature | Source | Computed via |
|---|---|---|
| `emotion_angry/happy/neutral/sad` | Stage 1 | `EmotionModel.predict()` (reused checkpoint) |
| `arousal`, `valence` | Stage 1 | same |
| `pitch_mean`, `pitch_std` | acoustic | `librosa.pyin` F0 |
| `energy_mean`, `energy_std` | acoustic | `librosa.feature.rms` |
| `speaking_rate` | acoustic | energy-envelope peak rate (syllable-nuclei proxy) |
| `speech_ratio` | acoustic (VAD) | energy-threshold voiced-frame ratio |
| `pause_count`, `pause_ratio` | acoustic | silence-segment detection |

**Not computed**: interruptions/overlap (review doc §5). Requires
dialog-level multi-channel audio + diarization; out of scope for a
single-utterance feature pipeline in the time available. Documented as a
known gap, not silently skipped.

## 4. Dataset & split

Reuses IEMOCAP and the exact same speaker-independent split as Stage 1
(Sessions 1-3 train / 4 val / 5 test) — no new dataset request, no new
licensing wait, and it's the same audio Stage 1 already validated against.

**Correction (found after the first real Kaggle run):** the first generated
notebook forgot to apply Stage 1's 4-class emotion filter (ang/hap+exc/sad/
neu only), so that run's feature tables cover all ~10,039 IEMOCAP utterances
per session instead of the ~5,533 Stage 1 actually validated on (train
5766/val 2103/test 2170 vs. Stage 1's 3260/1032/1241). This isn't fatal --
the extra utterances (raw labels like fru/sur/fea/dis/oth/xxx) still have
valid Stage-1 emotion/arousal/valence outputs and valid acoustic features,
and the CX bootstrap labels are computed from those outputs, not from the
true raw label, so nothing is mislabeled. It just means this run's model was
trained/evaluated on a broader utterance set than "identical to Stage 1"
claimed. The notebook generator (`_build_notebook.py`) has been fixed for
any future rerun; the models currently in `models/` are from the
**unfiltered** first run and were not retrained after the fix, to avoid
spending more Kaggle GPU time on a difference that doesn't affect validity.

## 5. Model

Four independent `XGBRegressor` models (stress, frustration, urgency,
escalation_risk) on the fused ~14-dim feature vector. Independent regressors
rather than one multi-output model: simpler, and lets each target's feature
importance be inspected separately (useful for the demo narrative — "here's
what's actually driving the urgency score").

Saved as `models/stress.json`, `models/frustration.json`,
`models/urgency.json`, `models/escalation_risk.json` (XGBoost's native
format — **not** `.pt`/`.pth`, that's a PyTorch-only extension and doesn't
apply here).

## 6. If/when EmoWork access is granted

Only the label source changes. `src`/notebook structure stays: swap the
bootstrap-formula label generation for a loader that reads EmoWork's real
self-reported stress/arousal labels, keep the same feature-fusion +
XGBoost training code. Frustration/urgency/escalation-risk would still need
their own justification since EmoWork only confirms stress + arousal +
valence, not all four targets (per review doc §90, don't assume a dataset
has every label).

## 7. Ready for Phase 2

`CXStateModel(stage1_checkpoint, stage2_models_dir).predict(wav_path)` will
be the stable interface (mirrors Stage 1's `EmotionModel`), returning
`{stress, frustration, urgency, escalation_risk}` — a direct, voice-grounded
replacement candidate for `agent/policy_engine.py`'s current text-based
`BehaviorSignals`. Integration itself is not implemented until the trained
models are handed back, per instruction.
