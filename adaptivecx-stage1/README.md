# AdaptiveCX — Stage 1 Independent Voice Emotion Model

A voice-based emotion intelligence model that works directly from the audio
waveform (not the transcript). Given a WAV file, it predicts:

1. Emotion (`angry` / `happy` / `neutral` / `sad`)
2. Arousal (continuous, `[-1, 1]`)
3. Valence (continuous, `[-1, 1]`)

This is an **independent project** (see `CLAUDE.md`): it does not depend on,
or get imported by, the existing AdaptiveCX LiveKit voice agent yet. It is
Phase 1 of a two-phase plan — build and validate the model standalone first,
integrate later.

```
WAV audio -> preprocessing -> emotion2vec+ (frozen) -> task heads -> Emotion + Arousal + Valence
```

## Model

- Backbone: `emotion2vec_plus_base`, loaded via the `funasr` library (its
  actual supported inference API — not plain `transformers.AutoModel`).
  - ModelScope id (tried first): `iic/emotion2vec_plus_base`
  - Hugging Face mirror (fallback): `emotion2vec/emotion2vec_plus_base`
- The backbone is **frozen** ("Stage A"). Three lightweight heads (a shared
  256-unit trunk + linear emotion/arousal/valence outputs) are trained on
  top of its embedding. See `src/model.py`.

## Dataset

**IEMOCAP**, attached via Kaggle (`sangayb/iemocap` mirror). Labels used:

- Emotion (categorical, from `EmoEvaluation` majority label)
- Arousal (IEMOCAP calls this "activation", 1-5 self-report scale)
- Valence (1-5 self-report scale)

4-class emotion mapping (documented choice, see `src/dataset.py`): `ang ->
angry`, `hap`/`exc -> happy`, `sad -> sad`, `neu -> neutral`. Other raw labels
(`fru`, `sur`, `fea`, `dis`, `oth`, `xxx`, ...) are dropped — low-frequency,
ambiguous, or non-emotion categories, not part of the target label set.

Arousal/valence are rescaled from IEMOCAP's 1-5 SAM scale to `[-1, 1]`.

**Speaker-independent split**: Sessions 1-3 = train (3260), Session 4 =
validation (1032), Session 5 = test (1241). IEMOCAP has 10 speakers, 2 per
session, disjoint across sessions, so this split has zero speaker leakage
(asserted in code, not just assumed).

## Training

Trained on Kaggle (GPU T4/P100) via `AdaptiveCX_Stage1_Kaggle.ipynb`
(generated from `_build_notebook.py`). Configuration (`configs/config.yaml`):

```yaml
model:
  name: iic/emotion2vec_plus_base
  freeze_backbone: true
training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 30
  seed: 42
  patience: 6
loss:
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0
```

Loss: `CrossEntropyLoss` (emotion) + `SmoothL1Loss` (arousal) + `SmoothL1Loss`
(valence), summed with the weights above. Early stopping on validation loss,
patience 6.

Because the backbone is frozen and never receives gradients, running it once
per file and caching the resulting embedding is mathematically equivalent to
re-running it every epoch, but far cheaper — so embeddings are extracted once
and heads are trained on the cached vectors (`src/dataset.py`).

## Evaluation (test / Session 5, 1241 utterances)

**Stage A (frozen backbone) — the model actually shipped, `models/best_stage1.pt`:**

| Emotion | | Arousal | | Valence | |
|---|---|---|---|---|---|
| Accuracy | 0.952 | MAE | 0.226 | MAE | 0.211 |
| Macro F1 | 0.956 | RMSE | 0.281 | RMSE | 0.276 |
| Weighted F1 | 0.952 | Pearson | 0.619 | Pearson | 0.834 |
| | | Spearman | 0.608 | Spearman | 0.835 |

Full numbers + confusion matrix: `results/evaluation.json`,
`results/confusion_matrix.png`.

**Stage B (partial backbone fine-tuning) — attempted, not shipped:**
per the project spec, Stage B (unfreezing the backbone's final layers) was
only worth trying because Stage A worked. It was tried and performed
substantially worse (emotion accuracy 0.446 vs. 0.952, valence Pearson 0.188
vs. 0.834 — see `results/evaluation_stageB.json`), most likely overfitting
the small unfrozen parameter set on this dataset size within the epoch
budget. **Stage A is the final model.** Stage B was not investigated further
per project decision (Phase-1 scope is "get a validated baseline," not
"exhaust every architecture variant").

## Limitations

- IEMOCAP is 10 actors performing scripted/improvised scenarios, not real
  customer-service audio; domain mismatch is expected at deployment time.
- The 4-class emotion mapping discards `fru`/`sur`/`fea`/`dis`/`oth`/`xxx`
  labels — this model cannot express those emotions even if present.
- Arousal/valence are 1-5 self-report annotations linearly rescaled to
  `[-1, 1]`; they are not a physiological or continuous ground truth.
- Stage A only: emotion2vec+ backbone is frozen; Stage B (partial
  fine-tuning) underperformed and was not pursued further.

## Project structure

```
adaptivecx-stage1/
├── data/
│   ├── raw/                 # (empty here — raw IEMOCAP lives on Kaggle)
│   ├── processed/           # cached embeddings, if regenerated locally
│   └── metadata/            # train/val/test_meta.csv (from the Kaggle run)
├── models/
│   └── best_stage1.pt       # shipped checkpoint (Stage A)
├── samples/
│   └── test.wav             # drop any WAV here to try prediction
├── src/                      # audio.py, dataset.py, model.py, train.py, evaluate.py, inference.py, utils.py
├── scripts/                  # prepare_data.py, train.py, evaluate.py, predict.py, check_audio.py
├── configs/config.yaml
├── results/                  # evaluation.json, confusion_matrix.png (Stage A) + evaluation_stageB.json
├── tests/
├── AdaptiveCX_Stage1_Kaggle.ipynb   # the notebook actually trained/run on Kaggle
├── _build_notebook.py               # generates the .ipynb above (source of truth for cell content)
├── requirements.txt
└── CLAUDE.md                        # the original project spec this was built against
```

## Setup (Windows PowerShell)

```powershell
cd adaptivecx-stage1
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell execution policy blocks activation, call the venv's Python
directly instead of activating: `.\.venv\Scripts\python.exe <script>`.

## Commands

```powershell
# Inspect a WAV file (sampling rate, channels, duration, sample count)
python scripts/check_audio.py --audio samples/test.wav

# Offline WAV inference against the shipped checkpoint (works without Kaggle
# or the dataset — only needs internet the very first time, to let funasr
# download and cache the emotion2vec+ checkpoint)
python scripts/predict.py --audio samples/test.wav

# Full local retrain (only if you have your own local IEMOCAP copy —
# training/evaluation of the shipped checkpoint happened on Kaggle):
python scripts/prepare_data.py --data-root <path to local IEMOCAP>
python scripts/train.py
python scripts/evaluate.py
```

## Ready for Phase 2

`EmotionModel(checkpoint_path, backbone_model, device, funasr_output_dir).predict(wav_path)`
(`src/inference.py`) is the stable interface the spec earmarks for the later
AdaptiveCX adapter layer:

```python
from stage1 import EmotionModel
model = EmotionModel("models/best.pt")
result = model.predict(audio)
```

Not implemented yet — per the spec, Phase 2 (live audio -> existing voice
agent -> Stage-1 inference -> existing dashboard) only starts after this
Phase-1 model is independently validated, which this README's evaluation
section documents.
