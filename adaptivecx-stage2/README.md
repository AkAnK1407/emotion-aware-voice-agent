# AdaptiveCX — Stage 2 Customer State Model

Predicts Stress / Frustration / Urgency / Escalation-risk from a WAV file, by
combining Stage 1's voice-emotion outputs with directly-computed acoustic
features (pitch, energy, speaking rate, pauses), fed into 4 XGBoost
regressors. Full design rationale, the label-availability problem, and the
documented bootstrap-label formula: see `CLAUDE_STAGE2.md`.

**Stage 1 is reused, not retrained.** This project only trains the 4 small
XGBoost heads on top of it.

## Status

Trained on Kaggle (`AdaptiveCX_Stage2_Kaggle.ipynb`) in two runs. Run 1
produced the acoustic-formula-labeled models (`models/{target}.json`); run 2
added the text-model-labeled variant (`models/{target}_textmodel.json`,
Section 9b — a free local GoEmotions-based classifier, no API key needed).
Results independently re-verified locally, not just trusted from the
Kaggle-side `evaluation.json` files — all numbers below were reproduced by
loading the downloaded models and re-predicting on the downloaded feature
CSVs.

## IMPORTANT: labels are a documented prototype scoring layer, not ground truth

No public dataset has real stress/frustration/urgency/escalation-risk labels
paired with customer-service voice on this project's timeline (see
`CLAUDE_STAGE2.md` §1 for the EmoWork/V2V investigation). The 4 targets are
computed by an explicit, documented formula (`CLAUDE_STAGE2.md` §2) from
Stage-1 + acoustic features — XGBoost is fit to *that formula*, not to
measured human ratings.

## Evaluation (test / Session 5, 2170 utterances)

### Acoustic-formula-labeled models (`models/{target}.json`)

| Target | MAE | RMSE | R² |
|---|---|---|---|
| stress | 0.025 | 0.029 | 0.964 |
| frustration | 0.005 | 0.006 | 0.999 |
| urgency | 0.012 | 0.014 | 0.982 |
| escalation_risk | 0.014 | 0.016 | 0.990 |

R² this high means XGBoost learned to approximate the deterministic formula
from the fused features — expected, since the formula *is* a function of
those same features (`arousal`, `valence`, emotion probs, `pitch_std`,
`speaking_rate`, `pause_ratio` are both the formula's inputs and the
model's). Not a measure of real-world accuracy. See `CLAUDE_STAGE2.md` §2.

Top features: **stress** → valence (0.72); **frustration** → angry-prob
(0.74); **urgency** → pause_ratio (0.42), speaking_rate (0.39);
**escalation_risk** → angry-prob (0.62), valence (0.21).

### Text-model-labeled models (`models/{target}_textmodel.json`)

| Target | MAE | RMSE | R² |
|---|---|---|---|
| stress | 0.041 | 0.065 | 0.065 |
| frustration | 0.065 | 0.092 | 0.088 |
| urgency | 0.043 | 0.069 | 0.073 |
| escalation_risk | 0.051 | 0.074 | 0.083 |

**Much lower R², and that's a real, informative result, not a bug.** This
variant's target labels come from GoEmotions reading the *transcript*
(word content), while the model's only inputs are *acoustic* features
(prosody + Stage-1 audio outputs). Low R² here means acoustic-only features
have limited power to predict what the words alone imply — i.e., **how
someone sounds and what they actually said only weakly agree** in this
data. Confirmed directly: Pearson correlation between the two label
sources on the same held-out utterances:

| Target | r (formula vs. text-model label) |
|---|---|
| stress | 0.24 |
| frustration | 0.27 |
| urgency | 0.11 |
| escalation_risk | 0.27 |

All weak-to-moderate, not strong. Also notable: the acoustic formula scores
utterances as more "stressed/frustrated" on average (mean ≈ 0.37-0.45)
than the text content does (mean ≈ 0.04-0.09) — consistent with acted
IEMOCAP dialogue often being delivered with dramatic/expressive prosody
even when the actual scripted words are mundane. Worth keeping in mind for
Phase 2: acoustic-only and content-aware CX scoring are picking up genuinely
different signals, not two noisy estimates of the same thing.

## Re-running efficiently (don't redo the slow step every time)

The notebook's feature-fusion step (Section 8) runs the full emotion2vec+
backbone across every utterance -- the same heavy compute Stage 1 itself
uses. Kaggle wipes `/kaggle/working` between sessions, so a fresh "Run All"
redoes that from scratch by default, which is what made the first run feel
like "it's running Stage 1 again." To skip it on later runs (e.g. when
adding the LLM-labeling section), attach `train_features.csv` /
`val_features.csv` / `test_features.csv` from `data/metadata/` in this
folder as a third Kaggle input dataset -- the notebook auto-detects them and
skips straight past the slow step.

## Known data-scope correction

The Kaggle run that produced these models used ~10,039 IEMOCAP utterances
(all raw emotion labels) rather than Stage 1's ~5,533-utterance 4-class
subset, due to a bug in the first generated notebook (fixed now in
`_build_notebook.py` for any future rerun, not retrained since it doesn't
affect validity of the current models — see `CLAUDE_STAGE2.md` §4).

## Files

```
adaptivecx-stage2/
├── models/                     # {target}.json (formula) + {target}_textmodel.json (text-model), XGBoost
├── data/metadata/               # train/val/test_features.csv + train/val/test_text_labels.csv
├── results/                     # evaluation.json, evaluation_textmodel.json, feature_importance.png
├── AdaptiveCX_Stage2_Kaggle.ipynb
├── _build_notebook.py
├── CLAUDE_STAGE2.md              # full design doc
└── README.md                     # this file
```

## Ready for Phase 2

`CXStateModel(stage1_checkpoint, stage2_models_dir).predict(wav_path) ->
{stress, frustration, urgency, escalation_risk}` (defined in the notebook,
Step 10) is the stable interface for integrating into
`agent/policy_engine.py`'s `BehaviorSignals`. Not implemented yet — pending
go-ahead.
