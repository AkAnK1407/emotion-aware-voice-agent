import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })

# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md("""# AdaptiveCX — Stage 1 Independent Voice Emotion Model

**Run this on Kaggle** (Settings → Accelerator: GPU T4/P100, Internet: ON).

Attach the IEMOCAP Kaggle dataset to this notebook (Add Input) before running.
The notebook auto-detects the dataset path under `/kaggle/input/`, so it does not
matter which exact IEMOCAP mirror/slug you attach, as long as it preserves the
standard `SessionX/dialog/EmoEvaluation/*.txt` + `SessionX/sentences/wav/**/*.wav`
layout.

## Scope (Phase 1 only, per project spec)

```
WAV audio -> preprocessing -> emotion2vec+ (frozen) -> task heads -> Emotion + Arousal + Valence
```

No LiveKit / WebSocket / LLM / TTS / dashboard integration here. This notebook is
self-contained and produces a checkpoint + evaluation report + a working WAV
inference call, which is the definition of "done" for Phase 1.

## Base model

`emotion2vec_plus_base`, loaded through the `funasr` library (its actual
supported inference API — not plain `transformers.AutoModel`):

- ModelScope id (default, tried first): `iic/emotion2vec_plus_base`
- Hugging Face mirror (fallback if ModelScope is unreachable): `emotion2vec/emotion2vec_plus_base`

`funasr.AutoModel(model=...)` downloads and caches the checkpoint automatically
on first use — no manual download step is required, as long as the Kaggle
notebook has Internet enabled.

## Dataset

IEMOCAP, attached as a Kaggle input dataset. Labels used:
- Emotion (categorical, from `EmoEvaluation` majority label)
- Arousal (IEMOCAP calls this "activation", 1-5 self-report scale)
- Valence (1-5 self-report scale)

Speaker-independent split: **Sessions 1-3 = train, Session 4 = validation,
Session 5 = test** (IEMOCAP has 10 unique speakers, 2 per session, disjoint
across sessions, so this split has zero speaker leakage).

## Design decision: precomputed embeddings (documented, not a spec deviation)

Stage A freezes emotion2vec+ and trains only the task heads. Because the
backbone is frozen and never receives gradients, running it once per file and
caching the resulting embedding is mathematically equivalent to re-running it
every epoch, but far cheaper on a Kaggle GPU/time budget. So: Section 7 below
extracts and caches one embedding per utterance; Sections 8+ train lightweight
heads on the cached embeddings.
""")

# ---------------------------------------------------------------------------
# 1. Installs
# ---------------------------------------------------------------------------
md("## 1. Install dependencies\n\nKaggle already ships torch/pandas/numpy/scikit-learn/matplotlib. We only need `funasr` (and its light dependencies) plus a couple of audio libs.")
code("""!pip install -q funasr modelscope soundfile librosa scipy tqdm
print("Install step complete.")
""")

# ---------------------------------------------------------------------------
# 2. Environment check (spec Step 1)
# ---------------------------------------------------------------------------
md("## 2. Step 1 — Inspect environment\n\nPer the project spec, report Python/PyTorch/CUDA/GPU before doing anything else.")
code("""import sys, torch, platform

print("Python      :", sys.version.split()[0])
print("Platform    :", platform.platform())
print("PyTorch     :", torch.__version__)
print("CUDA avail. :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
else:
    print("GPU         : none detected - enable GPU in Kaggle notebook settings")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
""")

# ---------------------------------------------------------------------------
# 3. Project directories (spec Step 2 / Section 17, condensed for Kaggle)
# ---------------------------------------------------------------------------
md("""## 3. Step 2 — Project directories

Kaggle's writable root is `/kaggle/working/`. We mirror the spec's
`models/`, `results/`, `data/processed/`, `samples/` layout under it.""")
code("""import os

BASE_DIR = "/kaggle/working/adaptivecx-stage1"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
EMB_CACHE_DIR = os.path.join(PROCESSED_DIR, "embeddings")

for d in (MODELS_DIR, RESULTS_DIR, PROCESSED_DIR, SAMPLES_DIR, EMB_CACHE_DIR):
    os.makedirs(d, exist_ok=True)

print("Project directories ready under:", BASE_DIR)
""")

# ---------------------------------------------------------------------------
# 4. Load & verify emotion2vec+ (spec Step 3 / Section 2)
# ---------------------------------------------------------------------------
md("""## 4. Step 3 — Load and verify emotion2vec+

Per the spec: verify the checkpoint loads, verify the inference API, verify the
embedding interface — before writing any training code. We try the ModelScope
id first, and fall back to the Hugging Face mirror if that is unreachable
(e.g. if Kaggle's Internet egress cannot reach modelscope.cn).""")
code("""from funasr import AutoModel

MODEL_ID_MODELSCOPE = "iic/emotion2vec_plus_base"
MODEL_ID_HF = "emotion2vec/emotion2vec_plus_base"

def load_emotion2vec():
    try:
        m = AutoModel(model=MODEL_ID_MODELSCOPE, hub="ms", disable_update=True)
        print(f"[MODEL] loaded '{MODEL_ID_MODELSCOPE}' from ModelScope")
        return m, MODEL_ID_MODELSCOPE, "ModelScope"
    except Exception as e:
        print(f"[MODEL] ModelScope load failed ({e!r}); falling back to Hugging Face hub")
        m = AutoModel(model=MODEL_ID_HF, hub="hf", disable_update=True)
        print(f"[MODEL] loaded '{MODEL_ID_HF}' from Hugging Face")
        return m, MODEL_ID_HF, "HuggingFace"

import time
_t0 = time.time()
emo2vec_model, EMO2VEC_CHECKPOINT_ID, EMO2VEC_SOURCE = load_emotion2vec()
print(f"[MODEL] load time = {time.time() - _t0:.2f}s")
""")

code("""# Verify the inference API + embedding interface on one real file.
# We do NOT assume the output dict shape - we print it so this is a verified
# fact about the installed funasr version, not an assumption.
import glob

EMB_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "_funasr_raw_out")
os.makedirs(EMB_OUTPUT_DIR, exist_ok=True)

def _find_any_wav(root="/kaggle/input"):
    for path in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        return path
    return None

_probe_wav = _find_any_wav()
assert _probe_wav is not None, "No .wav file found under /kaggle/input - attach the IEMOCAP dataset first."
print("[VERIFY] probing with:", _probe_wav)

_t0 = time.time()
_raw = emo2vec_model.generate(
    input=_probe_wav,
    granularity="utterance",
    extract_embedding=True,
    output_dir=EMB_OUTPUT_DIR,
)
print(f"[VERIFY] generate() latency = {time.time() - _t0:.2f}s")
print("[VERIFY] result type:", type(_raw))
print("[VERIFY] item keys  :", list(_raw[0].keys()))
for k, v in _raw[0].items():
    try:
        import numpy as np
        arr = np.asarray(v)
        print(f"           {k}: shape={arr.shape} dtype={arr.dtype}")
    except Exception:
        print(f"           {k}: {type(v)} = {v}")
""")

code("""# Defensive embedding extractor: works whether funasr returns the embedding
# inline under 'feats' or only writes it to output_dir as a .npy file
# (behavior has varied across funasr releases).
import numpy as np

def extract_embedding(wav_path, model, output_dir=EMB_OUTPUT_DIR):
    res = model.generate(
        input=wav_path,
        granularity="utterance",
        extract_embedding=True,
        output_dir=output_dir,
    )
    item = res[0]
    emb = item.get("feats", None)
    if emb is None:
        key = item.get("key", os.path.splitext(os.path.basename(wav_path))[0])
        candidates = glob.glob(os.path.join(output_dir, "**", f"{key}*.npy"), recursive=True)
        if candidates:
            emb = np.load(candidates[0])
    if emb is None:
        raise RuntimeError(
            f"Could not extract an embedding for {wav_path}; raw output keys={list(item.keys())}"
        )
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim > 1:
        # frame-level features - mean-pool over time to get one utterance vector
        emb = emb.mean(axis=0)
    return emb

_test_emb = extract_embedding(_probe_wav, emo2vec_model)
EMBED_DIM = _test_emb.shape[0]
print("[VERIFY] embedding dim (measured, not assumed):", EMBED_DIM)
""")

# ---------------------------------------------------------------------------
# 5. Locate IEMOCAP (spec Step 4)
# ---------------------------------------------------------------------------
md("""## 5. Step 4 — Locate and verify the IEMOCAP dataset

We search `/kaggle/input/` for the IEMOCAP signature (`dialog/EmoEvaluation` and
`sentences/wav`) instead of hardcoding a dataset slug, so this cell works
regardless of which specific Kaggle mirror you attached.""")
code("""def find_iemocap_root(search_root="/kaggle/input"):
    eval_dirs = glob.glob(os.path.join(search_root, "**", "dialog", "EmoEvaluation"), recursive=True)
    if not eval_dirs:
        raise FileNotFoundError(
            "Could not find an 'EmoEvaluation' directory under /kaggle/input. "
            "Attach the IEMOCAP dataset to this notebook (Add Input) and re-run."
        )
    # Each match's grandparent is a SessionX folder; the dataset root is one level above that.
    session_dirs = sorted({os.path.dirname(os.path.dirname(p)) for p in eval_dirs})
    dataset_roots = sorted({os.path.dirname(s) for s in session_dirs})
    return dataset_roots[0], session_dirs

IEMOCAP_ROOT, SESSION_DIRS = find_iemocap_root()
print("[DATA] IEMOCAP root:", IEMOCAP_ROOT)
print("[DATA] sessions found:")
for s in SESSION_DIRS:
    print("   -", s)
assert len(SESSION_DIRS) == 5, f"Expected 5 IEMOCAP sessions, found {len(SESSION_DIRS)}: {SESSION_DIRS}"
""")

# ---------------------------------------------------------------------------
# 6. Parse IEMOCAP metadata (spec Section 7 dataset adapter)
# ---------------------------------------------------------------------------
md("""## 6. Step 4b — Parse IEMOCAP labels into a metadata table

`EmoEvaluation/*.txt` lines look like:

```
[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
```

`[start - end]  utterance_id  emotion_label  [valence, activation, dominance]`

We parse only these summary lines (ignore per-annotator vote lines), resolve
each utterance id to its wav file under `sentences/wav/<dialog_id>/`, and
derive the speaker id from the id itself (`Ses01F_impro01_F000` -> session 01,
speaker F). Never invent labels: rows whose wav file cannot be found are
dropped and counted, not silently skipped.""")
code("""import re
import pandas as pd

LINE_RE = re.compile(
    r"^\\[([\\d.]+)\\s*-\\s*([\\d.]+)\\]\\s+(\\S+)\\s+(\\w+)\\s+\\[([\\d.]+),\\s*([\\d.]+),\\s*([\\d.]+)\\]"
)

def parse_session(session_dir):
    rows = []
    eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
    wav_root = os.path.join(session_dir, "sentences", "wav")
    txt_files = glob.glob(os.path.join(eval_dir, "*.txt"))
    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = LINE_RE.match(line.strip())
                if not m:
                    continue
                _, _, utt_id, emo_raw, val, act, dom = m.groups()
                dialog_id = utt_id.rsplit("_", 1)[0]
                wav_path = os.path.join(wav_root, dialog_id, utt_id + ".wav")
                session_num = utt_id[3:5]
                speaker_tag = utt_id.split("_")[-1][0]  # 'F' or 'M'
                speaker_id = f"Ses{session_num}_{speaker_tag}"
                rows.append({
                    "utt_id": utt_id,
                    "wav_path": wav_path,
                    "session": int(session_num),
                    "speaker_id": speaker_id,
                    "emotion_raw": emo_raw,
                    "valence_raw": float(val),
                    "arousal_raw": float(act),
                    "dominance_raw": float(dom),
                })
    return rows

all_rows = []
for sdir in SESSION_DIRS:
    all_rows.extend(parse_session(sdir))

meta = pd.DataFrame(all_rows)
print("[DATA] parsed utterances (pre wav-check):", len(meta))

meta["wav_exists"] = meta["wav_path"].apply(os.path.isfile)
n_missing = (~meta["wav_exists"]).sum()
if n_missing:
    print(f"[DATA] dropping {n_missing} rows with no matching wav file")
meta = meta[meta["wav_exists"]].drop(columns=["wav_exists"]).reset_index(drop=True)
print("[DATA] parsed utterances (with wav found):", len(meta))

print("\\n[DATA] raw emotion label distribution:")
print(meta["emotion_raw"].value_counts())

print("\\n[DATA] empirical valence range :", meta["valence_raw"].min(), "-", meta["valence_raw"].max())
print("[DATA] empirical arousal range :", meta["arousal_raw"].min(), "-", meta["arousal_raw"].max())
""")

code("""# Emotion label mapping -> 4-class (documented choice, not silent):
# ang -> angry, hap+exc -> happy, sad -> sad, neu -> neutral.
# All other raw labels (fru, sur, fea, dis, oth, xxx, ...) are dropped: they are
# either low-frequency, ambiguous, or non-emotion categories in IEMOCAP and are
# not part of the target label set defined in the project spec (Section 4).
EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
}
EMOTION_CLASSES = sorted(set(EMOTION_MAP.values()))  # -> ['angry', 'happy', 'neutral', 'sad']
EMOTION_TO_IDX = {c: i for i, c in enumerate(EMOTION_CLASSES)}

meta["emotion"] = meta["emotion_raw"].map(EMOTION_MAP)
n_before = len(meta)
meta = meta.dropna(subset=["emotion"]).reset_index(drop=True)
print(f"[DATA] kept {len(meta)} / {n_before} utterances after 4-class emotion mapping")
print("\\n[DATA] final class distribution:")
print(meta["emotion"].value_counts())

# Normalize IEMOCAP's empirically-confirmed 1-5 SAM scale to [-1, 1].
meta["valence"] = (meta["valence_raw"] - 3.0) / 2.0
meta["arousal"] = (meta["arousal_raw"] - 3.0) / 2.0
meta["emotion_idx"] = meta["emotion"].map(EMOTION_TO_IDX)

print("\\nEmotion classes:", EMOTION_CLASSES)
print("Arousal  : available: YES (normalized from IEMOCAP 'activation', range [-1, 1])")
print("Valence  : available: YES (normalized from IEMOCAP 'valence', range [-1, 1])")
""")

# ---------------------------------------------------------------------------
# 7. Speaker-independent split (spec Section 8)
# ---------------------------------------------------------------------------
md("""## 7. Step 4c — Speaker-independent split

Sessions 1-3 = train, Session 4 = validation, Session 5 = test. Each IEMOCAP
session contains 2 unique speakers who do not appear in any other session, so
splitting by session guarantees no speaker leakage.""")
code("""train_df = meta[meta["session"].isin([1, 2, 3])].reset_index(drop=True)
val_df = meta[meta["session"] == 4].reset_index(drop=True)
test_df = meta[meta["session"] == 5].reset_index(drop=True)

print("Dataset:")
print(f"  train:      {len(train_df)}")
print(f"  validation: {len(val_df)}")
print(f"  test:       {len(test_df)}")

train_speakers = set(train_df["speaker_id"])
val_speakers = set(val_df["speaker_id"])
test_speakers = set(test_df["speaker_id"])
overlap = (train_speakers & val_speakers) | (train_speakers & test_speakers) | (val_speakers & test_speakers)
print("\\nSpeaker independent split:", "YES" if not overlap else f"NO - overlap: {overlap}")
assert not overlap, "Speaker leakage detected across splits"
""")

# ---------------------------------------------------------------------------
# 8. Audio preprocessing utilities (spec Section 9 / Section 16 check_audio)
# ---------------------------------------------------------------------------
md("""## 8. Step 5 — Audio preprocessing utilities

Mono conversion, 16 kHz resampling, amplitude normalization, and
duration/NaN/empty-audio validation. Pauses are intentionally kept (no VAD
trimming) because prosody carries emotion information.""")
code("""import soundfile as sf
import librosa

TARGET_SR = 16000

def load_and_preprocess_audio(path, target_sr=TARGET_SR):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono conversion
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if wav.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if not np.isfinite(wav).all():
        raise ValueError(f"NaN/Inf samples in audio: {path}")
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak  # amplitude normalization
    return wav, sr

def inspect_audio(path):
    raw_wav, raw_sr = sf.read(path, always_2d=False)
    channels = 1 if raw_wav.ndim == 1 else raw_wav.shape[1]
    duration = len(raw_wav) / raw_sr
    print(f"Sampling rate   : {raw_sr}")
    print(f"Channels        : {channels}")
    print(f"Duration        : {duration:.3f}s")
    print(f"Number of samples: {len(raw_wav)}")
    return {"sampling_rate": raw_sr, "channels": channels, "duration": duration, "n_samples": len(raw_wav)}

# Sanity check on one real sample
_sanity_path = train_df.iloc[0]["wav_path"]
print("[CHECK] inspecting:", _sanity_path)
inspect_audio(_sanity_path)
_wav, _sr = load_and_preprocess_audio(_sanity_path)
_peak = np.abs(_wav).max()
print(f"[CHECK] preprocessed OK: shape={_wav.shape}, sr={_sr}, peak={_peak:.3f}")
""")

# ---------------------------------------------------------------------------
# 9. Feature extraction (frozen backbone, cached)
# ---------------------------------------------------------------------------
md("""## 9. Step 6 — Frozen-backbone feature extraction (cached)

Runs every utterance through emotion2vec+ once and caches the resulting
embedding to disk, keyed by `utt_id`. Resumable: already-cached files are
skipped, so a Kaggle session interruption does not lose progress. Files that
fail to load/extract are logged and dropped (not silently), matching the
spec's audio-validation requirement.""")
code("""from tqdm.auto import tqdm

def extract_split_embeddings(df, split_name):
    embeddings = np.zeros((len(df), EMBED_DIM), dtype=np.float32)
    keep_mask = np.ones(len(df), dtype=bool)
    failures = []

    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"extract[{split_name}]")):
        cache_path = os.path.join(EMB_CACHE_DIR, f"{row.utt_id}.npy")
        try:
            if os.path.isfile(cache_path):
                emb = np.load(cache_path)
            else:
                # Validate audio loads cleanly before spending compute on it.
                load_and_preprocess_audio(row.wav_path)
                emb = extract_embedding(row.wav_path, emo2vec_model)
                np.save(cache_path, emb)
            embeddings[i] = emb
        except Exception as e:
            keep_mask[i] = False
            failures.append((row.utt_id, str(e)))

    if failures:
        print(f"[DATA] {split_name}: {len(failures)} utterances failed extraction and were dropped")
        for utt_id, err in failures[:10]:
            print(f"    - {utt_id}: {err}")

    return embeddings[keep_mask], df[keep_mask].reset_index(drop=True)

train_emb, train_df = extract_split_embeddings(train_df, "train")
val_emb, val_df = extract_split_embeddings(val_df, "val")
test_emb, test_df = extract_split_embeddings(test_df, "test")

print("\\n[DATA] final embedding matrix shapes:")
print("  train:", train_emb.shape)
print("  val  :", val_emb.shape)
print("  test :", test_emb.shape)

np.save(os.path.join(PROCESSED_DIR, "train_embeddings.npy"), train_emb)
np.save(os.path.join(PROCESSED_DIR, "val_embeddings.npy"), val_emb)
np.save(os.path.join(PROCESSED_DIR, "test_embeddings.npy"), test_emb)
train_df.to_csv(os.path.join(PROCESSED_DIR, "train_meta.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, "val_meta.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test_meta.csv"), index=False)
print("[DATA] cached embeddings + metadata saved to", PROCESSED_DIR)
""")

# ---------------------------------------------------------------------------
# 10. Model architecture (spec Section 10)
# ---------------------------------------------------------------------------
md("""## 10. Step 6b — Model architecture: task heads

Simple shared trunk + three linear heads, as specified (no attention layers -
start simple). Arousal/valence outputs are passed through `tanh` since targets
are normalized to `[-1, 1]`.""")
code("""import torch.nn as nn

class EmotionHeads(nn.Module):
    def __init__(self, input_dim, num_emotions, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        emotion_logits = self.emotion_head(h)
        arousal = torch.tanh(self.arousal_head(h)).squeeze(-1)
        valence = torch.tanh(self.valence_head(h)).squeeze(-1)
        return emotion_logits, arousal, valence

print("EmotionHeads defined. input_dim will be", EMBED_DIM, "num_emotions will be", len(EMOTION_CLASSES))
""")

code("""class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, df):
        self.embeddings = embeddings
        self.emotion_idx = df["emotion_idx"].to_numpy()
        self.arousal = df["arousal"].to_numpy(dtype=np.float32)
        self.valence = df["valence"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.tensor(self.emotion_idx[idx], dtype=torch.long),
            torch.tensor(self.arousal[idx], dtype=torch.float32),
            torch.tensor(self.valence[idx], dtype=torch.float32),
        )

train_ds = EmbeddingDataset(train_emb, train_df)
val_ds = EmbeddingDataset(val_emb, val_df)
test_ds = EmbeddingDataset(test_emb, test_df)
print(f"Datasets ready: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
""")

# ---------------------------------------------------------------------------
# 11. Multi-task loss (spec Section 12)
# ---------------------------------------------------------------------------
md("## 11. Step 6c — Multi-task loss\n\n`L_total = lambda_emotion * L_emotion + lambda_arousal * L_arousal + lambda_valence * L_valence`, all weights configurable and defaulted to 1.0.")
code("""def multitask_loss(emotion_logits, arousal_pred, valence_pred,
                    emotion_label, arousal_label, valence_label, weights):
    ce = nn.CrossEntropyLoss()
    smooth_l1 = nn.SmoothL1Loss()

    l_emotion = ce(emotion_logits, emotion_label)
    l_arousal = smooth_l1(arousal_pred, arousal_label)
    l_valence = smooth_l1(valence_pred, valence_label)

    total = (
        weights["emotion"] * l_emotion
        + weights["arousal"] * l_arousal
        + weights["valence"] * l_valence
    )
    return total, {
        "emotion": l_emotion.item(),
        "arousal": l_arousal.item(),
        "valence": l_valence.item(),
    }
""")

# ---------------------------------------------------------------------------
# 12. Training (spec Section 11, 13, 21)
# ---------------------------------------------------------------------------
md("""## 12. Step 7-8 — Train (Stage A: frozen backbone)

Config mirrors the spec's suggested starting values. Backbone is already
frozen by construction (we only ever trained on its cached, detached
embeddings). Early stopping on validation loss; best checkpoint saved to
`models/best_stage1.pt`.""")
code("""from sklearn.metrics import f1_score, mean_absolute_error

CONFIG = {
    "model": {"name": EMO2VEC_CHECKPOINT_ID, "freeze_backbone": True},
    "training": {"batch_size": 8, "learning_rate": 1e-4, "epochs": 30, "seed": 42, "patience": 6},
    "loss": {"emotion_weight": 1.0, "arousal_weight": 1.0, "valence_weight": 1.0},
}

torch.manual_seed(CONFIG["training"]["seed"])
np.random.seed(CONFIG["training"]["seed"])

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

heads = EmotionHeads(input_dim=EMBED_DIM, num_emotions=len(EMOTION_CLASSES)).to(DEVICE)
optimizer = torch.optim.AdamW(heads.parameters(), lr=CONFIG["training"]["learning_rate"])
loss_weights = {
    "emotion": CONFIG["loss"]["emotion_weight"],
    "arousal": CONFIG["loss"]["arousal_weight"],
    "valence": CONFIG["loss"]["valence_weight"],
}

def run_epoch(loader, train_mode):
    heads.train(train_mode)
    total_loss = 0.0
    all_emotion_true, all_emotion_pred = [], []
    all_arousal_true, all_arousal_pred = [], []
    all_valence_true, all_valence_pred = [], []

    for emb, emo, aro, val in loader:
        emb, emo, aro, val = emb.to(DEVICE), emo.to(DEVICE), aro.to(DEVICE), val.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            emo_logits, aro_pred, val_pred = heads(emb)
            loss, _ = multitask_loss(emo_logits, aro_pred, val_pred, emo, aro, val, loss_weights)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * emb.size(0)
        all_emotion_true.extend(emo.cpu().numpy().tolist())
        all_emotion_pred.extend(emo_logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        all_arousal_true.extend(aro.cpu().numpy().tolist())
        all_arousal_pred.extend(aro_pred.detach().cpu().numpy().tolist())
        all_valence_true.extend(val.cpu().numpy().tolist())
        all_valence_pred.extend(val_pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "emotion_f1": f1_score(all_emotion_true, all_emotion_pred, average="macro", zero_division=0),
        "arousal_mae": mean_absolute_error(all_arousal_true, all_arousal_pred),
        "valence_mae": mean_absolute_error(all_valence_true, all_valence_pred),
    }
    return metrics

best_val_loss = float("inf")
epochs_without_improvement = 0
best_ckpt_path = os.path.join(MODELS_DIR, "best_stage1.pt")

for epoch in range(1, CONFIG["training"]["epochs"] + 1):
    train_metrics = run_epoch(train_loader, train_mode=True)
    val_metrics = run_epoch(val_loader, train_mode=False)

    print(f"Epoch {epoch}/{CONFIG['training']['epochs']}")
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Emotion F1: {val_metrics['emotion_f1']:.4f}")
    print(f"Arousal MAE: {val_metrics['arousal_mae']:.4f}")
    print(f"Valence MAE: {val_metrics['valence_mae']:.4f}")

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save({
            "model_state_dict": heads.state_dict(),
            "embed_dim": EMBED_DIM,
            "emotion_classes": EMOTION_CLASSES,
            "config": CONFIG,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }, best_ckpt_path)
        print("Best model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CONFIG["training"]["patience"]:
            print(f"\\nEarly stopping at epoch {epoch} (no improvement for {CONFIG['training']['patience']} epochs)")
            break
    print()

print("Training complete. Best checkpoint:", best_ckpt_path)
""")

# ---------------------------------------------------------------------------
# 13. Evaluation (spec Section 14, 22)
# ---------------------------------------------------------------------------
md("""## 13. Step 9 — Evaluation on held-out test set (Session 5)

Loads the best checkpoint and reports accuracy/F1/confusion matrix for emotion,
and MAE/RMSE/Pearson/Spearman for arousal and valence.""")
code("""import json
from sklearn.metrics import accuracy_score, f1_score as sk_f1, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
heads.load_state_dict(ckpt["model_state_dict"])
heads.eval()

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

emo_true, emo_pred = [], []
aro_true, aro_pred = [], []
val_true, val_pred = [], []

with torch.no_grad():
    for emb, emo, aro, val in test_loader:
        emb = emb.to(DEVICE)
        emo_logits, aro_p, val_p = heads(emb)
        emo_true.extend(emo.numpy().tolist())
        emo_pred.extend(emo_logits.argmax(dim=-1).cpu().numpy().tolist())
        aro_true.extend(aro.numpy().tolist())
        aro_pred.extend(aro_p.cpu().numpy().tolist())
        val_true.extend(val.numpy().tolist())
        val_pred.extend(val_p.cpu().numpy().tolist())

accuracy = accuracy_score(emo_true, emo_pred)
macro_f1 = sk_f1(emo_true, emo_pred, average="macro", zero_division=0)
weighted_f1 = sk_f1(emo_true, emo_pred, average="weighted", zero_division=0)
cm = confusion_matrix(emo_true, emo_pred, labels=list(range(len(EMOTION_CLASSES))))

aro_mae = mean_absolute_error(aro_true, aro_pred)
aro_rmse = float(np.sqrt(np.mean((np.array(aro_true) - np.array(aro_pred)) ** 2)))
aro_pearson = pearsonr(aro_true, aro_pred)[0] if len(set(aro_true)) > 1 else float("nan")
aro_spearman = spearmanr(aro_true, aro_pred)[0] if len(set(aro_true)) > 1 else float("nan")

val_mae = mean_absolute_error(val_true, val_pred)
val_rmse = float(np.sqrt(np.mean((np.array(val_true) - np.array(val_pred)) ** 2)))
val_pearson = pearsonr(val_true, val_pred)[0] if len(set(val_true)) > 1 else float("nan")
val_spearman = spearmanr(val_true, val_pred)[0] if len(set(val_true)) > 1 else float("nan")

print("Emotion")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print()
print("Arousal")
print(f"MAE: {aro_mae:.4f}")
print(f"RMSE: {aro_rmse:.4f}")
print(f"Pearson: {aro_pearson:.4f}")
print(f"Spearman: {aro_spearman:.4f}")
print()
print("Valence")
print(f"MAE: {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"Pearson: {val_pearson:.4f}")
print(f"Spearman: {val_spearman:.4f}")

evaluation_results = {
    "emotion": {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                "confusion_matrix": cm.tolist(), "classes": EMOTION_CLASSES},
    "arousal": {"mae": aro_mae, "rmse": aro_rmse, "pearson": aro_pearson, "spearman": aro_spearman},
    "valence": {"mae": val_mae, "rmse": val_rmse, "pearson": val_pearson, "spearman": val_spearman},
    "test_set_size": len(test_ds),
}
with open(os.path.join(RESULTS_DIR, "evaluation.json"), "w") as f:
    json.dump(evaluation_results, f, indent=2)

# Confusion matrix plot: single sequential hue (Blues), no rainbow colormap.
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(EMOTION_CLASSES)))
ax.set_yticks(range(len(EMOTION_CLASSES)))
ax.set_xticklabels(EMOTION_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(EMOTION_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Emotion confusion matrix (test / Session 5)")
vmax = cm.max() if cm.max() > 0 else 1
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > vmax * 0.6 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
fig.colorbar(im, ax=ax, label="count")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

print("\\nSaved:", os.path.join(RESULTS_DIR, "evaluation.json"))
print("Saved:", os.path.join(RESULTS_DIR, "confusion_matrix.png"))
""")

# ---------------------------------------------------------------------------
# 14. Inference (spec Section 15, 23, 24)
# ---------------------------------------------------------------------------
md("""## 14. Step 10 — WAV inference

`EmotionModel` wraps the frozen emotion2vec+ extractor + trained heads behind a
single `.predict(wav_path)` call, matching the interface the spec earmarks for
later Phase 2 integration (`from stage1 import EmotionModel`). Prints the exact
observability lines and result block the spec requires.""")
code("""class EmotionModel:
    def __init__(self, checkpoint_path, backbone_model, device=DEVICE):
        _t0 = time.time()
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.device = device
        self.emotion_classes = ckpt["emotion_classes"]
        self.heads = EmotionHeads(input_dim=ckpt["embed_dim"], num_emotions=len(self.emotion_classes)).to(device)
        self.heads.load_state_dict(ckpt["model_state_dict"])
        self.heads.eval()
        self.backbone = backbone_model
        print(f"[MODEL] loaded in {time.time() - _t0:.2f}s")

    def predict(self, wav_path):
        info = inspect_audio_quiet(wav_path)
        print(f"[AUDIO] duration={info['duration']:.2f}s")

        _t0 = time.time()
        load_and_preprocess_audio(wav_path)  # validation pass (mono/16k/NaN/empty checks)
        emb = extract_embedding(wav_path, self.backbone)
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emo_logits, aro_pred, val_pred = self.heads(emb_t)
            probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()[0]

        latency = time.time() - _t0
        pred_idx = int(probs.argmax())
        result = {
            "emotion": self.emotion_classes[pred_idx],
            "arousal": float(aro_pred.cpu().numpy()[0]),
            "valence": float(val_pred.cpu().numpy()[0]),
            "emotion_probabilities": {c: float(p) for c, p in zip(self.emotion_classes, probs)},
            "latency_sec": latency,
        }

        print(f"[INFERENCE] latency={latency:.2f}s")
        print(f"[RESULT] emotion={result['emotion']}")
        print(f"[RESULT] arousal={result['arousal']:.2f}")
        print(f"[RESULT] valence={result['valence']:.2f}")
        return result

def inspect_audio_quiet(path):
    raw_wav, raw_sr = sf.read(path, always_2d=False)
    return {"duration": len(raw_wav) / raw_sr}

def print_prediction_block(result):
    print("=" * 40)
    print("AdaptiveCX Stage 1")
    print("=" * 40)
    print()
    print(f"Emotion: {result['emotion']}")
    print(f"Arousal: {result['arousal']:.2f}")
    print(f"Valence: {result['valence']:.2f}")
    print()
    print("Emotion probabilities:")
    for cls, p in sorted(result["emotion_probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<8}: {p:.2f}")
    print()
    print(f"Inference latency: {result['latency_sec']:.2f} sec")
    print("=" * 40)
""")

code("""# Run real WAV inference (required before claiming the model works).
# Uses one held-out test-split file by default; replace SAMPLE_WAV with your
# own path (e.g. an uploaded samples/test.wav) to test any compatible WAV file.
SAMPLE_WAV = test_df.iloc[0]["wav_path"]

stage1_model = EmotionModel(best_ckpt_path, backbone_model=emo2vec_model)
result = stage1_model.predict(SAMPLE_WAV)
print()
print_prediction_block(result)
""")

# ---------------------------------------------------------------------------
# 15. Final report (spec Section 31)
# ---------------------------------------------------------------------------
md("## 15. Final report")
code("""print("1. Project structure   :", BASE_DIR)
print("2. Model checkpoint     :", EMO2VEC_CHECKPOINT_ID, f"(via {EMO2VEC_SOURCE})")
print("3. Dataset selected     : IEMOCAP -", IEMOCAP_ROOT)
print("4. Dataset labels       : emotion(4-class) + arousal + valence, normalized to [-1, 1]")
print("5. Training config      :", json.dumps(CONFIG, indent=2))
print("6. Evaluation results   :", json.dumps(evaluation_results, indent=2)[:800], "...")
print("7. Saved checkpoint     :", best_ckpt_path)
print("8. Kaggle setup         : attach IEMOCAP dataset as Input, enable GPU + Internet, Run All")
print("9. Data prep            : Sections 5-7 (auto-run as part of Run All)")
print("10. Training            : Section 12 (auto-run as part of Run All)")
print("11. Evaluation          : Section 13 (auto-run as part of Run All)")
print("12. WAV inference       : Section 14 - EmotionModel(...).predict(wav_path)")
print("13. Real inference output printed above (Section 14)")
print("14. Known limitations   :")
print("    - IEMOCAP is 10 actors performing scripted/improvised scenarios, not real")
print("      customer-service audio; domain mismatch is expected at deployment time.")
print("    - 4-class emotion mapping discards fru/sur/fea/dis/oth/xxx labels.")
print("    - Arousal/valence are 1-5 self-report annotations linearly rescaled to")
print("      [-1, 1]; they are not a physiological or continuous ground truth.")
print("    - Stage A only: emotion2vec+ backbone is frozen; no partial fine-tuning yet.")
print("15. Ready for Phase 2   : EmotionModel(checkpoint_path).predict(wav) is the")
print("    stable interface the spec earmarks for the later AdaptiveCX adapter layer.")
""")

# ---------------------------------------------------------------------------
# 16. Download results
# ---------------------------------------------------------------------------
md("""## 16. Download results

`/kaggle/working` is wiped when the session ends, so grab your results now.
This zips the checkpoint, evaluation metrics/plot, and split metadata CSVs
(not the raw embedding cache, to keep the file small), then prints a clickable
download link right in the notebook output. You can also get the same files
from the notebook's **Output** tab after **Save Version**.""")
code("""import shutil
from IPython.display import FileLink

DOWNLOAD_STAGE_DIR = "/kaggle/working/adaptivecx_download"
os.makedirs(DOWNLOAD_STAGE_DIR, exist_ok=True)

shutil.copytree(MODELS_DIR, os.path.join(DOWNLOAD_STAGE_DIR, "models"), dirs_exist_ok=True)
shutil.copytree(RESULTS_DIR, os.path.join(DOWNLOAD_STAGE_DIR, "results"), dirs_exist_ok=True)
for name in ("train_meta.csv", "val_meta.csv", "test_meta.csv"):
    src = os.path.join(PROCESSED_DIR, name)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(DOWNLOAD_STAGE_DIR, name))

zip_base = "/kaggle/working/adaptivecx_stage1_results"
zip_path = shutil.make_archive(zip_base, "zip", DOWNLOAD_STAGE_DIR)
print("Zipped:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.2f} MB)")

# Relative to /kaggle/working, which is the notebook's cwd - click to download.
FileLink(os.path.relpath(zip_path, "/kaggle/working"))
""")

# ---------------------------------------------------------------------------
# 17. Stage B - partial backbone fine-tuning (optional, spec Section 11 Stage B)
# ---------------------------------------------------------------------------
md("""## 17. Stage B — optional partial fine-tuning

**Read this before running.** Stage A used `funasr`'s public `generate()` API,
which is a stable, documented inference path. Stage B needs gradients to flow
into the last few backbone layers, which `generate()` does not expose - so this
section reaches into the underlying `torch.nn.Module` (`emo2vec_model.model`)
directly. That internal object's exact structure isn't part of funasr's public
API and can vary between versions, so every risky assumption below is checked
live instead of hardcoded:

1. We work on a **deep copy** of the backbone, so Stage A's model object (and
   therefore Sections 13-14's results/inference) stay untouched no matter what
   happens here or what order you run cells in.
2. We auto-discover the transformer layer stack instead of guessing an
   attribute name, and print what we found.
3. Before training, we run one file through our own differentiable forward
   path and compare it against Stage A's `generate()`-based embedding for the
   *same* file (cosine similarity) - if these don't closely agree, the
   discovered forward path is wrong and training would be trained on
   garbage. **Do not proceed past that check if it fails; inspect the printed
   architecture and adjust `backbone_forward` first.**

Only the last `UNFREEZE_LAST_N` transformer blocks get `requires_grad=True`;
everything else (conv feature extractor, embeddings, earlier layers) stays
frozen, with a much smaller learning rate than the heads, per the spec's
"small learning rate" instruction for Stage B.""")

code("""import copy

STAGE_B_CONFIG = {
    "unfreeze_last_n_layers": 2,
    "head_lr": 1e-4,
    "backbone_lr": 1e-6,
    "epochs": 5,
    "patience": 2,
    "grad_accum_steps": 8,  # batch_size=1 physically (variable-length audio, no padding) + accumulation
}

stageB_backbone = copy.deepcopy(emo2vec_model.model).to(DEVICE)
print("Backbone cloned for Stage B. Type:", type(stageB_backbone))
""")

code("""# Discover the transformer block stack generically: look for any ModuleList
# with 2+ entries. Do not assume a specific attribute name.
def find_transformer_stacks(module):
    return [(name, sub) for name, sub in module.named_modules()
            if isinstance(sub, nn.ModuleList) and len(sub) >= 2]

stack_candidates = find_transformer_stacks(stageB_backbone)
print("ModuleList candidates found in the backbone:")
for name, ml in stack_candidates:
    print(f"  {name}: {len(ml)} blocks")

assert stack_candidates, (
    "No ModuleList found in the backbone. Print(stageB_backbone) below and "
    "manually identify the transformer layer stack before continuing."
)

TARGET_STACK_NAME, TARGET_LAYERS = max(stack_candidates, key=lambda kv: len(kv[1]))
print(f"\\nSelected stack for partial unfreezing: '{TARGET_STACK_NAME}' ({len(TARGET_LAYERS)} blocks)")
""")

code("""# Defensive forward: try the common FunASR/wav2vec2-style entry points in
# order and use whichever one actually works, instead of assuming one.
BACKBONE_FORWARD_METHOD = None

def backbone_forward(model, wav_tensor):
    global BACKBONE_FORWARD_METHOD
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)

    attempts = [BACKBONE_FORWARD_METHOD] if BACKBONE_FORWARD_METHOD else ["extract_features", "forward_features", "__call__"]
    out, last_err = None, None
    used = None
    for attempt in attempts:
        if attempt is None:
            continue
        try:
            fn = model if attempt == "__call__" else getattr(model, attempt)
            out = fn(wav_tensor)
            used = attempt
            break
        except Exception as e:
            last_err = e
    if out is None:
        raise RuntimeError(f"All backbone forward attempts failed. Last error: {last_err!r}")
    if BACKBONE_FORWARD_METHOD is None:
        BACKBONE_FORWARD_METHOD = used
        print(f"[STAGE B] backbone forward call that works: model.{used}(wav)")

    if isinstance(out, dict):
        for key in ("x", "features", "last_hidden_state", "hidden_states"):
            if key in out and torch.is_tensor(out[key]):
                out = out[key]
                break
        else:
            out = next(v for v in out.values() if torch.is_tensor(v))
    elif isinstance(out, (tuple, list)):
        out = out[0]

    if out.dim() == 3:
        pooled = out.mean(dim=1)       # (1, T, H) -> (1, H)
    elif out.dim() == 2:
        pooled = out                   # already pooled -> (1, H)
    else:
        raise RuntimeError(f"Unexpected backbone output shape: {tuple(out.shape)}")
    return pooled.squeeze(0)
""")

code("""# EQUIVALENCE CHECK - do not skip. Confirms our differentiable forward path
# agrees with Stage A's generate()-based embedding for the same file, before
# we trust it enough to train on.
stageB_backbone.eval()
_probe_wav_arr, _ = load_and_preprocess_audio(_sanity_path)
_probe_tensor = torch.from_numpy(_probe_wav_arr).to(DEVICE)

with torch.no_grad():
    _manual_emb = backbone_forward(stageB_backbone, _probe_tensor).cpu().numpy()

_reference_emb = extract_embedding(_sanity_path, emo2vec_model)

_cos_sim = float(
    np.dot(_manual_emb, _reference_emb)
    / (np.linalg.norm(_manual_emb) * np.linalg.norm(_reference_emb) + 1e-8)
)
print(f"[VERIFY] cosine similarity (manual forward vs generate()): {_cos_sim:.4f}")
if _cos_sim < 0.95:
    print("[VERIFY] WARNING: low agreement. Inspect stageB_backbone's architecture below")
    print("         and adjust backbone_forward's pooling/output-key logic before training.")
    print(stageB_backbone)
else:
    print("[VERIFY] OK - manual forward path matches the verified inference path closely enough to train on.")
""")

code("""# Freeze everything, then unfreeze only the last N blocks of the discovered stack.
for p in stageB_backbone.parameters():
    p.requires_grad = False

for layer in TARGET_LAYERS[-STAGE_B_CONFIG["unfreeze_last_n_layers"]:]:
    for p in layer.parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in stageB_backbone.parameters() if p.requires_grad)
total = sum(p.numel() for p in stageB_backbone.parameters())
print(f"Backbone trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
""")

# ---------------------------------------------------------------------------
# 18. Stage B data + training
# ---------------------------------------------------------------------------
md("""## 18. Stage B — raw-audio dataset and training loop

Fine-tuning needs gradients through the backbone every step, so we can't reuse
the cached embeddings from Stage A here - this loads raw waveforms on the fly.
Batches are size 1 (utterances have different lengths and we don't know if
this backbone's forward path supports attention-mask padding) with gradient
accumulation to get an effective batch size, and the heads are warm-started
from the Stage A checkpoint rather than randomly initialized.""")
code("""class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths = df["wav_path"].tolist()
        self.emotion_idx = df["emotion_idx"].to_numpy()
        self.arousal = df["arousal"].to_numpy(dtype=np.float32)
        self.valence = df["valence"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, _ = load_and_preprocess_audio(self.paths[idx])
        return (
            torch.from_numpy(wav),
            torch.tensor(self.emotion_idx[idx], dtype=torch.long),
            torch.tensor(self.arousal[idx], dtype=torch.float32),
            torch.tensor(self.valence[idx], dtype=torch.float32),
        )

train_raw_loader = torch.utils.data.DataLoader(RawAudioDataset(train_df), batch_size=1, shuffle=True)
val_raw_loader = torch.utils.data.DataLoader(RawAudioDataset(val_df), batch_size=1, shuffle=False)
test_raw_loader = torch.utils.data.DataLoader(RawAudioDataset(test_df), batch_size=1, shuffle=False)

heads_b = EmotionHeads(input_dim=EMBED_DIM, num_emotions=len(EMOTION_CLASSES)).to(DEVICE)
heads_b.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE)["model_state_dict"])
print("Stage B heads warm-started from Stage A checkpoint:", best_ckpt_path)

optimizer_b = torch.optim.AdamW([
    {"params": heads_b.parameters(), "lr": STAGE_B_CONFIG["head_lr"]},
    {"params": [p for p in stageB_backbone.parameters() if p.requires_grad], "lr": STAGE_B_CONFIG["backbone_lr"]},
])
""")

code("""def run_epoch_stage_b(loader, train_mode):
    # Always eval() the backbone: wav2vec2/data2vec-style SSL backbones like
    # emotion2vec+ can apply pretraining-only behavior (e.g. span masking)
    # when in .train() mode, which the frozen-inference path never exercised.
    # requires_grad (not .train()/.eval()) is what controls whether the
    # unfrozen layers still receive gradients, so this does not block fine-tuning.
    stageB_backbone.eval()
    heads_b.train(train_mode)
    accum = STAGE_B_CONFIG["grad_accum_steps"]
    total_loss = 0.0
    emo_true, emo_pred, aro_true, aro_pred, val_true, val_pred = [], [], [], [], [], []

    if train_mode:
        optimizer_b.zero_grad()

    for step, (wav, emo, aro, val) in enumerate(tqdm(loader, desc="stageB", leave=False)):
        wav = wav.squeeze(0).to(DEVICE)
        emo, aro, val = emo.to(DEVICE), aro.to(DEVICE), val.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            emb = backbone_forward(stageB_backbone, wav).unsqueeze(0)
            emo_logits, aro_p, val_p = heads_b(emb)
            loss, _ = multitask_loss(emo_logits, aro_p, val_p, emo, aro, val, loss_weights)
            if train_mode:
                (loss / accum).backward()
                if (step + 1) % accum == 0:
                    optimizer_b.step()
                    optimizer_b.zero_grad()

        total_loss += loss.item()
        emo_true.append(emo.item()); emo_pred.append(int(emo_logits.argmax(dim=-1).item()))
        aro_true.append(aro.item()); aro_pred.append(float(aro_p.item()))
        val_true.append(val.item()); val_pred.append(float(val_p.item()))

    if train_mode and (len(loader) % accum != 0):
        optimizer_b.step()
        optimizer_b.zero_grad()

    return {
        "loss": total_loss / len(loader),
        "emotion_f1": f1_score(emo_true, emo_pred, average="macro", zero_division=0),
        "arousal_mae": mean_absolute_error(aro_true, aro_pred),
        "valence_mae": mean_absolute_error(val_true, val_pred),
    }

best_val_loss_b = float("inf")
epochs_without_improvement_b = 0
best_ckpt_path_b = os.path.join(MODELS_DIR, "best_stageB.pt")

for epoch in range(1, STAGE_B_CONFIG["epochs"] + 1):
    train_metrics_b = run_epoch_stage_b(train_raw_loader, train_mode=True)
    val_metrics_b = run_epoch_stage_b(val_raw_loader, train_mode=False)

    print(f"Epoch {epoch}/{STAGE_B_CONFIG['epochs']}")
    print(f"Train Loss: {train_metrics_b['loss']:.4f}")
    print(f"Val Loss: {val_metrics_b['loss']:.4f}")
    print(f"Emotion F1: {val_metrics_b['emotion_f1']:.4f}")
    print(f"Arousal MAE: {val_metrics_b['arousal_mae']:.4f}")
    print(f"Valence MAE: {val_metrics_b['valence_mae']:.4f}")

    if val_metrics_b["loss"] < best_val_loss_b:
        best_val_loss_b = val_metrics_b["loss"]
        epochs_without_improvement_b = 0
        torch.save({
            "heads_state_dict": heads_b.state_dict(),
            "backbone_state_dict": stageB_backbone.state_dict(),
            "embed_dim": EMBED_DIM,
            "emotion_classes": EMOTION_CLASSES,
            "stage_b_config": STAGE_B_CONFIG,
            "unfrozen_stack_name": TARGET_STACK_NAME,
            "backbone_forward_method": BACKBONE_FORWARD_METHOD,
            "epoch": epoch,
            "val_metrics": val_metrics_b,
        }, best_ckpt_path_b)
        print("Best Stage B model saved.")
    else:
        epochs_without_improvement_b += 1
        if epochs_without_improvement_b >= STAGE_B_CONFIG["patience"]:
            print(f"\\nEarly stopping at epoch {epoch}")
            break
    print()

print("Stage B training complete. Best checkpoint:", best_ckpt_path_b)
""")

# ---------------------------------------------------------------------------
# 19. Stage B evaluation + comparison
# ---------------------------------------------------------------------------
md("## 19. Stage B — evaluation on the same held-out test set (Session 5), compared against Stage A")
code("""ckpt_b = torch.load(best_ckpt_path_b, map_location=DEVICE)
heads_b.load_state_dict(ckpt_b["heads_state_dict"])
stageB_backbone.load_state_dict(ckpt_b["backbone_state_dict"])
heads_b.eval()
stageB_backbone.eval()

emo_true, emo_pred, aro_true, aro_pred, val_true, val_pred = [], [], [], [], [], []
with torch.no_grad():
    for wav, emo, aro, val in tqdm(test_raw_loader, desc="stageB eval", leave=False):
        wav = wav.squeeze(0).to(DEVICE)
        emb = backbone_forward(stageB_backbone, wav).unsqueeze(0)
        emo_logits, aro_p, val_p = heads_b(emb)
        emo_true.append(emo.item()); emo_pred.append(int(emo_logits.argmax(dim=-1).item()))
        aro_true.append(aro.item()); aro_pred.append(float(aro_p.item()))
        val_true.append(val.item()); val_pred.append(float(val_p.item()))

accuracy_b = accuracy_score(emo_true, emo_pred)
macro_f1_b = sk_f1(emo_true, emo_pred, average="macro", zero_division=0)
weighted_f1_b = sk_f1(emo_true, emo_pred, average="weighted", zero_division=0)

aro_mae_b = mean_absolute_error(aro_true, aro_pred)
aro_rmse_b = float(np.sqrt(np.mean((np.array(aro_true) - np.array(aro_pred)) ** 2)))
aro_pearson_b = pearsonr(aro_true, aro_pred)[0] if len(set(aro_true)) > 1 else float("nan")
aro_spearman_b = spearmanr(aro_true, aro_pred)[0] if len(set(aro_true)) > 1 else float("nan")

val_mae_b = mean_absolute_error(val_true, val_pred)
val_rmse_b = float(np.sqrt(np.mean((np.array(val_true) - np.array(val_pred)) ** 2)))
val_pearson_b = pearsonr(val_true, val_pred)[0] if len(set(val_true)) > 1 else float("nan")
val_spearman_b = spearmanr(val_true, val_pred)[0] if len(set(val_true)) > 1 else float("nan")

evaluation_results_b = {
    "emotion": {"accuracy": accuracy_b, "macro_f1": macro_f1_b, "weighted_f1": weighted_f1_b},
    "arousal": {"mae": aro_mae_b, "rmse": aro_rmse_b, "pearson": aro_pearson_b, "spearman": aro_spearman_b},
    "valence": {"mae": val_mae_b, "rmse": val_rmse_b, "pearson": val_pearson_b, "spearman": val_spearman_b},
    "test_set_size": len(test_raw_loader.dataset),
}
with open(os.path.join(RESULTS_DIR, "evaluation_stageB.json"), "w") as f:
    json.dump(evaluation_results_b, f, indent=2)

print(f"{'metric':<18}{'Stage A':>12}{'Stage B':>12}")
print(f"{'emotion acc':<18}{evaluation_results['emotion']['accuracy']:>12.4f}{accuracy_b:>12.4f}")
print(f"{'emotion macro F1':<18}{evaluation_results['emotion']['macro_f1']:>12.4f}{macro_f1_b:>12.4f}")
print(f"{'arousal MAE':<18}{evaluation_results['arousal']['mae']:>12.4f}{aro_mae_b:>12.4f}")
print(f"{'arousal pearson':<18}{evaluation_results['arousal']['pearson']:>12.4f}{aro_pearson_b:>12.4f}")
print(f"{'valence MAE':<18}{evaluation_results['valence']['mae']:>12.4f}{val_mae_b:>12.4f}")
print(f"{'valence pearson':<18}{evaluation_results['valence']['pearson']:>12.4f}{val_pearson_b:>12.4f}")
print("\\n(Lower MAE is better. Higher accuracy/F1/pearson is better.)")
print("Saved:", os.path.join(RESULTS_DIR, "evaluation_stageB.json"))
""")

# ---------------------------------------------------------------------------
# 20. Stage B inference
# ---------------------------------------------------------------------------
md("## 20. Stage B — WAV inference with the fine-tuned backbone")
code("""def predict_stage_b(wav_path):
    info = inspect_audio_quiet(wav_path)
    print(f"[AUDIO] duration={info['duration']:.2f}s")

    _t0 = time.time()
    wav_arr, _ = load_and_preprocess_audio(wav_path)
    wav_t = torch.from_numpy(wav_arr).to(DEVICE)

    with torch.no_grad():
        emb = backbone_forward(stageB_backbone, wav_t).unsqueeze(0)
        emo_logits, aro_p, val_p = heads_b(emb)
        probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()[0]

    latency = time.time() - _t0
    pred_idx = int(probs.argmax())
    result = {
        "emotion": EMOTION_CLASSES[pred_idx],
        "arousal": float(aro_p.cpu().numpy()[0]),
        "valence": float(val_p.cpu().numpy()[0]),
        "emotion_probabilities": {c: float(p) for c, p in zip(EMOTION_CLASSES, probs)},
        "latency_sec": latency,
    }
    print(f"[INFERENCE] latency={latency:.2f}s")
    print(f"[RESULT] emotion={result['emotion']}")
    print(f"[RESULT] arousal={result['arousal']:.2f}")
    print(f"[RESULT] valence={result['valence']:.2f}")
    return result

result_b = predict_stage_b(SAMPLE_WAV)
print()
print_prediction_block(result_b)
""")

# ---------------------------------------------------------------------------
# 21. Download Stage B results
# ---------------------------------------------------------------------------
md("## 21. Download Stage B results")
code("""shutil.copy(best_ckpt_path_b, os.path.join(DOWNLOAD_STAGE_DIR, "models", "best_stageB.pt"))
shutil.copy(os.path.join(RESULTS_DIR, "evaluation_stageB.json"),
            os.path.join(DOWNLOAD_STAGE_DIR, "results", "evaluation_stageB.json"))

zip_path_b = shutil.make_archive(zip_base, "zip", DOWNLOAD_STAGE_DIR)
print("Zipped:", zip_path_b, f"({os.path.getsize(zip_path_b) / 1e6:.2f} MB)")
FileLink(os.path.relpath(zip_path_b, "/kaggle/working"))
""")

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "AdaptiveCX_Stage1_Kaggle.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {out_path} with {len(cells)} cells")
