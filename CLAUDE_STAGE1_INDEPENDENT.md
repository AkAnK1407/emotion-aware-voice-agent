# CLAUDE.md — AdaptiveCX Independent Stage-1 Voice Emotion Model

## 0. IMPORTANT: BUILD THIS INDEPENDENTLY FIRST

This is an **independent ML project**.

Do NOT integrate it with the existing AdaptiveCX voice agent yet.

The existing AdaptiveCX project will only be integrated **after this standalone model is trained, evaluated, and successfully tested on WAV files**.

### Development strategy

```text
PHASE 1 — Independent ML Model
        ↓
Dataset
        ↓
Audio preprocessing
        ↓
emotion2vec+
        ↓
Emotion + Arousal + Valence
        ↓
Training
        ↓
Evaluation
        ↓
WAV inference
        ↓
MODEL VALIDATED
        ↓
PHASE 2 — AdaptiveCX Integration
        ↓
Live audio
        ↓
Existing voice agent
        ↓
Dashboard / WebSocket
```

For now, ONLY implement Phase 1.

Do not add:
- LiveKit integration
- WebSocket integration
- LLM integration
- TTS integration
- existing AdaptiveCX dashboard integration
- Stage-2 customer-state model
- production infrastructure

The purpose is to get a clean, independently testable Stage-1 model working first.

---

# 1. OBJECTIVE

Build a voice-based emotion intelligence model that works directly from the **audio waveform**, not from the transcript.

The model should produce:

1. Emotion classification
2. Arousal score
3. Valence score

Target architecture:

```text
                    WAV AUDIO
                        │
                        ▼
              Audio Preprocessing
              - mono
              - 16 kHz
              - normalization
              - optional VAD
                        │
                        ▼
                  emotion2vec+
                        │
                        ▼
                Speech Embedding
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Emotion        Arousal       Valence
     Classification  Regression    Regression
          │             │             │
          ▼             ▼             ▼
       Emotion       0–1 / dataset   dataset
       probability      scale         scale
```

The exact output ranges MUST be determined from the selected training dataset.

Do not assume the range.

---

# 2. PRIMARY MODEL

Use the Hugging Face model:

`emotion2vec/emotion2vec_plus_base`

Before implementation:

1. Verify that the checkpoint currently exists.
2. Verify the recommended inference API.
3. Verify its embedding/output interface.
4. Verify compatibility with the current Python/PyTorch environment.
5. Document the exact checkpoint used.

If the checkpoint cannot be used because of compatibility/API problems:

- investigate the official emotion2vec implementation/checkpoint;
- consider a WavLM fallback;
- explain the reason for the fallback.

Do NOT silently switch to an unrelated model.

---

# 3. WHY emotion2vec+

The model is specifically designed for speech emotion representation.

We want to use it as the pretrained acoustic representation and add our own downstream task heads.

IMPORTANT:

Do not claim that the pretrained checkpoint automatically gives:

```text
Emotion + Arousal + Valence
```

unless this is actually verified.

Our intended architecture is:

```text
emotion2vec+
      │
      ▼
Speech embedding
      │
      ├── Emotion classification head
      ├── Arousal regression head
      └── Valence regression head
```

The downstream heads must be trained using appropriate labeled data.

---

# 4. DATASET REQUIREMENTS

We need data for:

## Task A — Emotion classification

Example labels:

```text
angry
happy
sad
neutral
```

The exact class list must come from the selected dataset.

## Task B — Arousal regression

A numerical arousal label.

## Task C — Valence regression

A numerical valence label.

---

# 5. DATASET SELECTION

Investigate the following candidates:

### IEMOCAP

Potentially useful for:

- emotion classification
- arousal
- valence
- possibly dominance

### MSP-Podcast

Potentially useful for:

- naturalistic speech emotion
- arousal
- valence
- dominance

### RAVDESS

Use mainly as a quick baseline if needed.

Important limitation:

RAVDESS contains acted emotional speech and may not represent natural customer conversations well.

---

# 6. DATASET DECISION RULE

Before writing the training code, inspect the candidate datasets and determine:

```text
Dataset
├── Audio available?
├── Emotion labels?
├── Arousal labels?
├── Valence labels?
├── Speaker IDs?
├── Number of samples?
├── License/access?
└── Suitable for our three tasks?
```

Select the best practical dataset for the 5-day prototype.

Do not spend time downloading and preprocessing multiple datasets unless necessary.

If no single dataset provides all three tasks cleanly:

1. Clearly document the limitation.
2. Prefer a dataset that supports arousal + valence if those are the main research outputs.
3. Use a separate emotion dataset only if necessary.
4. Keep the data adapters separate.

Do not merge datasets blindly.

---

# 7. DATASET ADAPTER

Create:

```text
src/dataset.py
```

Responsibilities:

- metadata loading
- audio loading
- label mapping
- arousal normalization
- valence normalization
- train/validation/test splitting
- speaker-independent splitting

Return something like:

```python
{
    "audio": waveform,
    "sampling_rate": 16000,
    "emotion_label": emotion,
    "arousal": arousal,
    "valence": valence
}
```

If a sample does not contain a particular label, represent it as missing and mask that task's loss.

Never invent labels.

---

# 8. SPEAKER-INDEPENDENT SPLITTING

This is important.

If speaker IDs are available, do NOT randomly split individual audio files into train/test.

Instead:

```text
Train speakers
Validation speakers
Test speakers
```

This reduces speaker leakage and gives a more meaningful evaluation.

Document the exact split.

---

# 9. AUDIO PREPROCESSING

Create:

```text
src/audio.py
```

Implement:

- WAV/audio loading
- mono conversion
- 16 kHz resampling
- amplitude normalization
- optional VAD
- duration validation
- NaN/empty-audio validation

Do not aggressively remove pauses because prosody is important for emotion.

The preprocessing pipeline should be deterministic where possible.

---

# 10. MODEL ARCHITECTURE

Create:

```text
src/model.py
```

Architecture:

```text
              Audio
                │
                ▼
          emotion2vec+
                │
                ▼
        Shared embedding
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Emotion  Arousal  Valence
       Head     Head      Head
```

### Emotion head

Use:

```text
Linear/MLP
→ number of emotion classes
→ logits
```

### Arousal head

Use:

```text
Linear/MLP
→ 1 value
```

### Valence head

Use:

```text
Linear/MLP
→ 1 value
```

Start simple.

Do not add unnecessary attention layers unless there is a clear reason.

---

# 11. TRAINING STRATEGY

Use two stages.

## Stage A — Frozen backbone

Initially:

```text
emotion2vec+ = frozen
task heads = trainable
```

This should be the first working implementation.

Advantages:

- much faster
- lower GPU requirement
- easier debugging
- suitable for a 5-day prototype

## Stage B — Optional partial fine-tuning

Only if Stage A works:

```text
freeze most emotion2vec+ layers
unfreeze final layers
small learning rate
```

Compare performance.

Do not fine-tune the entire model unless necessary.

---

# 12. MULTI-TASK LOSS

Implement:

```text
L_total =
    λ_emotion * L_emotion
  + λ_arousal * L_arousal
  + λ_valence * L_valence
```

Initial configuration:

```text
λ_emotion = 1.0
λ_arousal = 1.0
λ_valence = 1.0
```

Make these configurable.

Use:

```text
Emotion → CrossEntropyLoss
Arousal → MSELoss or SmoothL1Loss
Valence → MSELoss or SmoothL1Loss
```

Mask missing labels.

---

# 13. TRAINING SCRIPT

Create:

```text
src/train.py
```

Support:

- CPU
- CUDA if available
- batch size
- learning rate
- epochs
- random seed
- checkpoint saving
- best-model saving
- early stopping if useful
- training loss
- validation loss

Suggested configuration:

```yaml
model:
  name: emotion2vec/emotion2vec_plus_base
  freeze_backbone: true

training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 10
  seed: 42

loss:
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0
```

These are starting values, not mandatory final values.

Tune only if necessary.

---

# 14. EVALUATION

Create:

```text
src/evaluate.py
```

### Emotion

Calculate:

- Accuracy
- Macro F1
- Weighted F1
- Confusion matrix

### Arousal

Calculate:

- MAE
- RMSE
- Pearson correlation if practical
- Spearman correlation if practical

### Valence

Calculate:

- MAE
- RMSE
- Pearson correlation if practical
- Spearman correlation if practical

Save:

```text
results/evaluation.json
results/confusion_matrix.png
```

---

# 15. INFERENCE

Create:

```text
src/inference.py
```

The first required inference mode is OFFLINE WAV inference.

Example:

```powershell
python scripts/predict.py --audio samples/test.wav
```

Output:

```text
========================================
AdaptiveCX Stage 1
========================================

Emotion: angry
Arousal: 0.82
Valence: -0.61

Emotion probabilities:
  angry   : 0.82
  happy   : 0.04
  sad     : 0.09
  neutral : 0.05

Inference latency: 0.43 sec
========================================
```

These values are examples only.

Never hard-code them.

---

# 16. SAMPLE AUDIO TESTING

Create:

```text
samples/
```

The project must support:

```text
samples/test.wav
```

The user should be able to replace this with any compatible WAV file.

Also provide a command to inspect audio:

```powershell
python scripts/check_audio.py --audio samples/test.wav
```

Print:

```text
Sampling rate
Channels
Duration
Number of samples
```

---

# 17. PROJECT STRUCTURE

Create this independent project:

```text
adaptivecx-stage1/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│
├── samples/
│   └── test.wav
│
├── src/
│   ├── __init__.py
│   ├── audio.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── check_audio.py
│
├── configs/
│   └── config.yaml
│
├── results/
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

Do not depend on the existing AdaptiveCX repository.

This project should run independently.

---

# 18. REQUIREMENTS

Create:

```text
requirements.txt
```

Keep it minimal.

Likely packages:

```text
torch
torchaudio
transformers
datasets
numpy
pandas
scikit-learn
librosa
soundfile
pyyaml
matplotlib
tqdm
```

Only add packages actually required by the selected emotion2vec implementation.

Before installing:

```powershell
python --version
```

Prefer Python 3.12 compatibility.

---

# 19. WINDOWS POWERSHELL SETUP

The user uses Windows PowerShell.

Provide exact commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell execution policy causes activation problems, provide an alternative command rather than changing system policy unnecessarily.

---

# 20. DATA PREPARATION COMMAND

Create:

```powershell
python scripts/prepare_data.py
```

It should:

- verify dataset path
- validate metadata
- validate audio
- resample/prepare if required
- generate train/validation/test metadata
- print dataset statistics

Example:

```text
Dataset:
  train:  XXXX
  validation: XXX
  test: XXX

Emotion classes:
  angry
  happy
  sad
  neutral

Arousal:
  available: YES

Valence:
  available: YES

Speaker independent split: YES
```

Use the actual dataset information.

---

# 21. TRAIN COMMAND

Create:

```powershell
python scripts/train.py
```

The script should:

1. load config
2. load dataset
3. initialize emotion2vec+
4. initialize task heads
5. train
6. evaluate validation set
7. save best checkpoint

Example output:

```text
Epoch 1/10
Train Loss: ...
Val Loss: ...

Emotion F1: ...
Arousal MAE: ...
Valence MAE: ...

Best model saved.
```

---

# 22. EVALUATION COMMAND

Create:

```powershell
python scripts/evaluate.py
```

Output:

```text
Emotion
Accuracy: ...
Macro F1: ...

Arousal
MAE: ...
RMSE: ...

Valence
MAE: ...
RMSE: ...
```

---

# 23. PREDICTION COMMAND

Create:

```powershell
python scripts/predict.py --audio samples/test.wav
```

This must work WITHOUT the existing AdaptiveCX application.

The only required input is a WAV file and the trained model.

---

# 24. OBSERVABILITY FOR THIS INDEPENDENT PROJECT

Keep this simple.

Record:

```text
audio duration
model load time
inference latency
predicted emotion
arousal
valence
```

Example:

```text
[MODEL] loaded in 1.82s
[AUDIO] duration=3.42s
[INFERENCE] latency=0.41s
[RESULT] emotion=angry
[RESULT] arousal=0.82
[RESULT] valence=-0.61
```

No database or production observability system is required.

---

# 25. TESTS

Create:

```text
tests/
```

Test:

1. audio loading
2. resampling
3. dataset loading
4. model initialization
5. model forward pass
6. output dimensions
7. inference
8. JSON output
9. acoustic/audio validation

Example expected output dimensions:

```text
emotion_logits = [batch_size, num_emotions]
arousal = [batch_size, 1]
valence = [batch_size, 1]
```

---

# 26. RESEARCH DOCUMENTATION

Create:

```text
README.md
```

Include:

### Model

- emotion2vec+ checkpoint
- why it was selected
- exact version/checkpoint used

### Dataset

- dataset name
- source
- license/access
- number of samples
- labels
- split strategy

### Architecture

```text
Audio
 ↓
Preprocessing
 ↓
emotion2vec+
 ↓
Embedding
 ├── Emotion
 ├── Arousal
 └── Valence
```

### Training

- loss
- optimizer
- learning rate
- batch size
- epochs
- frozen/unfrozen layers

### Evaluation

- metrics
- results

### Limitations

Explicitly state:

- acted vs natural speech if applicable
- dataset limitations
- domain mismatch with customer service
- limitations of arousal/valence prediction
- model size/latency

---

# 27. RESEARCH REFERENCES

Verify before documenting:

1. “emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation”
2. Hugging Face `emotion2vec/emotion2vec_plus_base`
3. “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing”
4. “Adapting WavLM for Speech Emotion Recognition”
5. “Attention-Augmented End-to-End Multi-Task Learning for Emotion Prediction from Speech”

Do not fabricate publication details or claims.

---

# 28. IMPORTANT DEVELOPMENT ORDER

Do NOT implement everything simultaneously.

Follow this exact order.

## STEP 1 — Inspect environment

Check:

```powershell
python --version
pip --version
```

Check whether CUDA is available.

Report:

```text
Python
PyTorch
CUDA
GPU
```

## STEP 2 — Create project structure

Only independent files.

## STEP 3 — Verify emotion2vec+

Before writing training code:

- download/load checkpoint
- run one WAV
- verify the API
- verify embedding extraction

## STEP 4 — Verify dataset

Confirm:

- audio
- emotion
- arousal
- valence
- speaker information

## STEP 5 — Build preprocessing

Test one sample.

## STEP 6 — Build model

Run a single forward pass.

## STEP 7 — Train emotion baseline

Get one working training run.

## STEP 8 — Add arousal + valence

Only if the selected data supports them.

## STEP 9 — Evaluate

Generate metrics.

## STEP 10 — WAV inference

Run:

```powershell
python scripts/predict.py --audio samples/test.wav
```

## STEP 11 — Improve if necessary

Only after the baseline works.

Possible improvements:

- partial backbone fine-tuning
- class weighting
- augmentation
- better pooling
- hyperparameter tuning

Do not start with these.

---

# 29. DEFINITION OF "MODEL WORKS"

The independent project is considered complete when:

```text
Dataset
   ↓
Training
   ↓
Checkpoint
   ↓
Evaluation
   ↓
WAV
   ↓
Inference
   ↓
Emotion + Arousal + Valence
```

all work without depending on AdaptiveCX.

The user should be able to clone/copy this folder onto another machine, install requirements, and reproduce the experiment.

---

# 30. PHASE 2 — ONLY AFTER PHASE 1 WORKS

Do NOT implement this now.

Later, create a separate integration layer:

```text
adaptivecx-stage1/
        │
        │ model checkpoint
        ▼
AdaptiveCX Adapter
        │
        ▼
Existing voice agent
        │
        ▼
Live audio
        │
        ▼
Stage-1 inference
        │
        ▼
Emotion + Arousal + Valence
        │
        ▼
Existing dashboard
```

The integration layer should NOT modify the core ML code unnecessarily.

Prefer an API/class like:

```python
from stage1 import EmotionModel

model = EmotionModel("models/best.pt")

result = model.predict(audio)
```

The exact interface can be designed after the independent model works.

---

# 31. FINAL OUTPUT REQUIRED FROM CLAUDE

After implementation, report:

```text
1. Project structure
2. Model checkpoint
3. Dataset selected
4. Dataset labels
5. Training configuration
6. Evaluation results
7. Saved checkpoint location
8. Exact Windows installation command
9. Exact data preparation command
10. Exact training command
11. Exact evaluation command
12. Exact WAV inference command
13. Example REAL inference output
14. Known limitations
15. What is ready for later AdaptiveCX integration
```

IMPORTANT:

Do not tell the user that the model is working until an actual WAV inference has been executed successfully.

---

# 32. MOST IMPORTANT RULE

The first milestone is NOT:

```text
Live voice agent
```

The first milestone is:

```text
                    test.wav
                       │
                       ▼
                preprocessing
                       │
                       ▼
                  emotion2vec+
                       │
                       ▼
                  task heads
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Emotion   Arousal   Valence
             │         │         │
             └─────────┼─────────┘
                       ▼
                     JSON
```

Make this work first.

Only then integrate it into AdaptiveCX.
