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
md("""# AdaptiveCX — Stage 2 Customer State Model

**Run this on Kaggle** (Settings -> Accelerator: GPU T4/P100, Internet: ON).

Attach these Kaggle inputs before running:
1. The IEMOCAP dataset (same one used for Stage 1).
2. Your Stage 1 results -- upload `best_stage1.pt` (from
   `adaptivecx-stage1/models/best_stage1.pt` on your machine) as a new
   Kaggle Dataset (Add Data -> New Dataset -> upload the file), then attach
   it here. The notebook auto-detects it under `/kaggle/input/` by filename.
3. **Optional, but attach this every time after your first successful run:**
   your downloaded `train_features.csv` / `val_features.csv` /
   `test_features.csv` (from `adaptivecx-stage2/data/metadata/` on your
   machine), uploaded as a third Kaggle Dataset. This skips Step 6 below --
   the slow part that runs the full emotion2vec+ backbone across every
   utterance (the same heavy compute Stage 1 itself uses) -- so re-running
   this notebook to add/change something later (like the text-model-labeling
   section) takes minutes instead of the full backbone pass again. This is
   what "it feels like it's running Stage 1 again" was: Stage 1's *checkpoint*
   isn't retrained, but its *backbone* does get re-run across every
   utterance every time this step isn't skipped.

## Scope (per `CLAUDE_STAGE2.md` / the team's ML & CX Flow Review doc)

```
Stage 1 outputs (emotion, arousal, valence)
        +
Acoustic features (pitch, energy, speaking rate, pauses)
        =
Fused feature vector -> XGBoost -> Stress, Frustration, Urgency, Escalation risk
```

Stage 1 is **not retrained** here -- its checkpoint is loaded and reused as a
frozen feature source, exactly as the review doc specifies ("Stage 1 outputs"
is one of Stage 2's three input groups).

## The label problem, addressed honestly

No public dataset pairs real customer-service voice with real stress /
frustration / urgency / escalation-risk labels on this project's timeline
(EmoWork is a real, well-matched dataset but is gated behind a Data Use
Agreement + application process; see `CLAUDE_STAGE2.md` section 1 for the
full writeup). Per the review doc's own explicit fallback, this notebook
uses a **documented, transparent bootstrap/prototype scoring formula**
(section 2 of `CLAUDE_STAGE2.md`) as the training target -- not a claim of
real ground truth. Evaluation metrics below measure how well XGBoost fits
that documented formula, not real-world accuracy against true customer
state. This is stated here once, plainly, and is not hidden anywhere else
in this notebook.

## Dataset

IEMOCAP, same speaker-independent split as Stage 1 (Sessions 1-3 = train,
4 = validation, 5 = test) -- reused, not re-requested.
""")

# ---------------------------------------------------------------------------
# 1. Installs
# ---------------------------------------------------------------------------
md("## 1. Install dependencies")
code("""!pip install -q funasr modelscope soundfile librosa scipy tqdm xgboost scikit-learn matplotlib pandas numpy
print("Install step complete.")""")

# ---------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------
md("## 2. Step 1 — Inspect environment")
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
print("Using device:", DEVICE)""")

# ---------------------------------------------------------------------------
# 3. Project dirs
# ---------------------------------------------------------------------------
md("## 3. Step 2 — Project directories")
code("""import os

BASE_DIR = "/kaggle/working/adaptivecx-stage2"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EMB_CACHE_DIR = os.path.join(PROCESSED_DIR, "embeddings")

for d in (MODELS_DIR, RESULTS_DIR, PROCESSED_DIR, EMB_CACHE_DIR):
    os.makedirs(d, exist_ok=True)

print("Project directories ready under:", BASE_DIR)""")

# ---------------------------------------------------------------------------
# 4. Load Stage 1 backbone + heads
# ---------------------------------------------------------------------------
md("""## 4. Step 3 — Load Stage 1 (frozen backbone + trained heads)

Stage 1 is reused, not retrained. The backbone (emotion2vec+) is loaded the
same way Stage 1 loaded it; the heads checkpoint is *your* already-trained
`best_stage1.pt`, located automatically under `/kaggle/input/`.""")
code("""from funasr import AutoModel
import glob, time
import torch.nn as nn

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

_t0 = time.time()
emo2vec_model, EMO2VEC_CHECKPOINT_ID, EMO2VEC_SOURCE = load_emotion2vec()
print(f"[MODEL] backbone load time = {time.time() - _t0:.2f}s")

# Locate your uploaded Stage 1 checkpoint.
_ckpt_candidates = glob.glob("/kaggle/input/**/best_stage1.pt", recursive=True)
assert _ckpt_candidates, (
    "Could not find best_stage1.pt under /kaggle/input. Upload "
    "adaptivecx-stage1/models/best_stage1.pt as a Kaggle Dataset and attach it."
)
STAGE1_CKPT_PATH = _ckpt_candidates[0]
print("[MODEL] found Stage 1 checkpoint:", STAGE1_CKPT_PATH)

class EmotionHeads(nn.Module):
    def __init__(self, input_dim, num_emotions, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        emotion_logits = self.emotion_head(h)
        arousal = torch.tanh(self.arousal_head(h)).squeeze(-1)
        valence = torch.tanh(self.valence_head(h)).squeeze(-1)
        return emotion_logits, arousal, valence

_stage1_ckpt = torch.load(STAGE1_CKPT_PATH, map_location=DEVICE)
EMOTION_CLASSES = _stage1_ckpt["emotion_classes"]
EMBED_DIM = _stage1_ckpt["embed_dim"]
stage1_heads = EmotionHeads(input_dim=EMBED_DIM, num_emotions=len(EMOTION_CLASSES)).to(DEVICE)
stage1_heads.load_state_dict(_stage1_ckpt["model_state_dict"])
stage1_heads.eval()
print("[MODEL] Stage 1 heads loaded. Emotion classes:", EMOTION_CLASSES, "| embed_dim:", EMBED_DIM)""")

# ---------------------------------------------------------------------------
# 5. Embedding extraction (same defensive extractor as Stage 1)
# ---------------------------------------------------------------------------
md("## 5. Step 3b — Embedding extraction helper (same approach as Stage 1)")
code("""import numpy as np

EMB_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "_funasr_raw_out")
os.makedirs(EMB_OUTPUT_DIR, exist_ok=True)

def extract_embedding(wav_path, model, output_dir=EMB_OUTPUT_DIR):
    res = model.generate(input=wav_path, granularity="utterance", extract_embedding=True, output_dir=output_dir)
    item = res[0]
    emb = item.get("feats", None)
    if emb is None:
        key = item.get("key", os.path.splitext(os.path.basename(wav_path))[0])
        candidates = glob.glob(os.path.join(output_dir, "**", f"{key}*.npy"), recursive=True)
        if candidates:
            emb = np.load(candidates[0])
    if emb is None:
        raise RuntimeError(f"Could not extract an embedding for {wav_path}; raw output keys={list(item.keys())}")
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim > 1:
        emb = emb.mean(axis=0)
    return emb

@torch.no_grad()
def stage1_predict(wav_path):
    \"\"\"Real Stage-1 forward pass: emotion probabilities + arousal + valence.\"\"\"
    emb = extract_embedding(wav_path, emo2vec_model)
    emb_t = torch.from_numpy(emb).unsqueeze(0).to(DEVICE)
    emo_logits, aro, val = stage1_heads(emb_t)
    probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()[0]
    return {
        **{f"emotion_{c}": float(p) for c, p in zip(EMOTION_CLASSES, probs)},
        "arousal": float(aro.cpu().numpy()[0]),
        "valence": float(val.cpu().numpy()[0]),
    }

print("stage1_predict() ready.")""")

# ---------------------------------------------------------------------------
# 6. IEMOCAP parsing (reused from Stage 1)
# ---------------------------------------------------------------------------
md("## 6. Step 4 — Locate + parse IEMOCAP (identical to Stage 1)")
code("""def find_iemocap_root(search_root="/kaggle/input"):
    eval_dirs = glob.glob(os.path.join(search_root, "**", "dialog", "EmoEvaluation"), recursive=True)
    if not eval_dirs:
        raise FileNotFoundError("Could not find an 'EmoEvaluation' directory under /kaggle/input. Attach IEMOCAP.")
    session_dirs = sorted({os.path.dirname(os.path.dirname(p)) for p in eval_dirs})
    dataset_roots = sorted({os.path.dirname(s) for s in session_dirs})
    return dataset_roots[0], session_dirs

IEMOCAP_ROOT, SESSION_DIRS = find_iemocap_root()
print("[DATA] IEMOCAP root:", IEMOCAP_ROOT)
assert len(SESSION_DIRS) == 5, f"Expected 5 IEMOCAP sessions, found {len(SESSION_DIRS)}"

import re
import pandas as pd

LINE_RE = re.compile(r"^\\[([\\d.]+)\\s*-\\s*([\\d.]+)\\]\\s+(\\S+)\\s+(\\w+)\\s+\\[([\\d.]+),\\s*([\\d.]+),\\s*([\\d.]+)\\]")

def parse_session(session_dir):
    rows = []
    eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
    wav_root = os.path.join(session_dir, "sentences", "wav")
    for txt_path in glob.glob(os.path.join(eval_dir, "*.txt")):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = LINE_RE.match(line.strip())
                if not m:
                    continue
                _, _, utt_id, emo_raw, val, act, dom = m.groups()
                dialog_id = utt_id.rsplit("_", 1)[0]
                wav_path = os.path.join(wav_root, dialog_id, utt_id + ".wav")
                session_num = utt_id[3:5]
                speaker_tag = utt_id.split("_")[-1][0]
                rows.append({
                    "utt_id": utt_id, "wav_path": wav_path, "session": int(session_num),
                    "speaker_id": f"Ses{session_num}_{speaker_tag}", "emotion_raw": emo_raw,
                    "valence_raw": float(val), "arousal_raw": float(act), "dominance_raw": float(dom),
                })
    return rows

all_rows = []
for sdir in SESSION_DIRS:
    all_rows.extend(parse_session(sdir))
meta = pd.DataFrame(all_rows)
meta["wav_exists"] = meta["wav_path"].apply(os.path.isfile)
meta = meta[meta["wav_exists"]].drop(columns=["wav_exists"]).reset_index(drop=True)
print("[DATA] utterances with wav found:", len(meta))

# Same 4-class filter as Stage 1 (ang/hap+exc/sad/neu only) -- keeps this
# notebook's utterance set identical to Stage 1's, since Stage 1's emotion
# head was only ever trained/validated on these 4 classes.
_EMOTION_MAP = {"ang": "angry", "hap": "happy", "exc": "happy", "sad": "sad", "neu": "neutral"}
meta["emotion_raw_mapped"] = meta["emotion_raw"].map(_EMOTION_MAP)
_n_before = len(meta)
meta = meta.dropna(subset=["emotion_raw_mapped"]).drop(columns=["emotion_raw_mapped"]).reset_index(drop=True)
print(f"[DATA] kept {len(meta)} / {_n_before} utterances after applying Stage 1's 4-class filter")

train_df = meta[meta["session"].isin([1, 2, 3])].reset_index(drop=True)
val_df = meta[meta["session"] == 4].reset_index(drop=True)
test_df = meta[meta["session"] == 5].reset_index(drop=True)
print(f"Dataset: train={len(train_df)} val={len(val_df)} test={len(test_df)}")""")

# ---------------------------------------------------------------------------
# 7. Acoustic feature extraction
# ---------------------------------------------------------------------------
md("""## 7. Step 5 — Acoustic feature extraction (review doc §5)

Pitch, energy, speaking rate, VAD speech ratio, and pause structure --
computed directly from the waveform, no separate model needed. Interruptions
/ overlap are **not** computed here (documented gap, see `CLAUDE_STAGE2.md`
section 3: needs dialog-level multi-channel audio + diarization).""")
code("""import librosa
import soundfile as sf

TARGET_SR = 16000

def extract_acoustic_features(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    duration = len(wav) / sr
    if duration <= 0 or not np.isfinite(wav).all():
        raise ValueError(f"Invalid audio: {wav_path}")

    # Pitch (F0) via pYIN, voiced frames only.
    f0, voiced_flag, _ = librosa.pyin(wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.nanmean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.nanstd(voiced_f0)) if voiced_f0.size else 0.0

    # Energy (RMS).
    rms = librosa.feature.rms(y=wav)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # VAD: energy-threshold voiced-frame ratio (simple, deterministic, no extra model).
    thresh = max(energy_mean * 0.5, 1e-4)
    voiced_frames = rms > thresh
    speech_ratio = float(np.mean(voiced_frames))

    # Speaking rate proxy: peaks in the smoothed RMS envelope ~= syllable nuclei.
    from scipy.signal import find_peaks
    smoothed = np.convolve(rms, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smoothed, height=thresh, distance=3)
    speaking_rate = float(len(peaks) / duration)

    # Pauses: contiguous runs of non-voiced frames.
    hop_s = 512 / sr  # librosa default hop_length
    pause_count, cur_run = 0, 0
    for v in voiced_frames:
        if not v:
            cur_run += 1
        else:
            if cur_run > 0:
                pause_count += 1
            cur_run = 0
    if cur_run > 0:
        pause_count += 1
    pause_ratio = float(1.0 - speech_ratio)

    return {
        "pitch_mean": pitch_mean, "pitch_std": pitch_std,
        "energy_mean": energy_mean, "energy_std": energy_std,
        "speech_ratio": speech_ratio, "speaking_rate": speaking_rate,
        "pause_count": float(pause_count), "pause_ratio": pause_ratio,
        "duration": duration,
    }

# Sanity check on one real file.
_probe = train_df.iloc[0]["wav_path"]
print("[CHECK] acoustic features for:", _probe)
print(extract_acoustic_features(_probe))""")

# ---------------------------------------------------------------------------
# 8. Fuse features across splits
# ---------------------------------------------------------------------------
md("""## 8. Step 6 — Fuse Stage-1 outputs + acoustic features (cached per utterance)

This is the one GPU/CPU-heavy pass: runs the real Stage-1 model plus the
acoustic pipeline on every utterance, once, and caches to disk so re-running
this cell is instant on a second pass.""")
code("""from tqdm.auto import tqdm

def build_feature_table(df, split_name):
    cache_path = os.path.join(PROCESSED_DIR, f"{split_name}_features.csv")

    # This is the expensive step -- it runs the full emotion2vec+ backbone
    # (the same heavy compute Stage 1 uses) across every utterance, plus
    # acoustic feature extraction. /kaggle/working is wiped between Kaggle
    # sessions, so a fresh "Run All" always redoes this from scratch unless
    # you attach a PREVIOUS run's {split}_features.csv as a Kaggle input
    # dataset (upload the file you already downloaded, same way you did for
    # best_stage1.pt) -- then it's found here and the whole extraction pass
    # is skipped.
    input_hits = glob.glob(f"/kaggle/input/**/{split_name}_features.csv", recursive=True)
    if input_hits:
        print(f"[DATA] {split_name}: found precomputed features in attached input {input_hits[0]}, skipping extraction")
        return pd.read_csv(input_hits[0])
    if os.path.isfile(cache_path):
        print(f"[DATA] {split_name}: loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    failures = []
    for row in tqdm(df.itertuples(), total=len(df), desc=f"features[{split_name}]"):
        try:
            feat = {"utt_id": row.utt_id, "session": row.session, "speaker_id": row.speaker_id}
            feat.update(stage1_predict(row.wav_path))
            feat.update(extract_acoustic_features(row.wav_path))
            rows.append(feat)
        except Exception as e:
            failures.append((row.utt_id, str(e)))

    if failures:
        print(f"[DATA] {split_name}: {len(failures)} utterances failed and were dropped")
        for utt_id, err in failures[:10]:
            print(f"    - {utt_id}: {err}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cache_path, index=False)
    print(f"[DATA] {split_name}: {len(out_df)} feature rows -> {cache_path}")
    return out_df

train_feat = build_feature_table(train_df, "train")
val_feat = build_feature_table(val_df, "val")
test_feat = build_feature_table(test_df, "test")""")

# ---------------------------------------------------------------------------
# 9. Bootstrap CX labels
# ---------------------------------------------------------------------------
md("""## 9. Step 7 — Bootstrap CX-state labels (documented formula, NOT ground truth)

See `CLAUDE_STAGE2.md` section 2 for the full rationale. Printed here in
full so the formula is never hidden from anyone reading notebook output.""")
code("""def normalize01(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-8:
        return series * 0.0
    return (series - lo) / (hi - lo)

def add_bootstrap_labels(feat_df):
    df = feat_df.copy()
    neg_emotion = df["emotion_angry"] + df["emotion_sad"]
    high_arousal = (df["arousal"] + 1) / 2
    neg_valence = (1 - df["valence"]) / 2
    pitch_var = normalize01(df["pitch_std"].fillna(0))
    speak_rate = normalize01(df["speaking_rate"])
    energy_level = normalize01(df["energy_mean"])
    pause_ratio = df["pause_ratio"]

    df["stress"] = (0.35 * neg_emotion + 0.30 * high_arousal + 0.20 * pitch_var + 0.15 * (1 - pause_ratio)).clip(0, 1)
    df["frustration"] = (0.40 * df["emotion_angry"] + 0.25 * high_arousal + 0.20 * neg_valence + 0.15 * speak_rate).clip(0, 1)
    df["urgency"] = (0.40 * speak_rate + 0.30 * high_arousal + 0.20 * (1 - pause_ratio) + 0.10 * energy_level).clip(0, 1)
    df["escalation_risk"] = (0.5 * df["stress"] + 0.5 * df["frustration"]).clip(0, 1)
    return df

train_feat = add_bootstrap_labels(train_feat)
val_feat = add_bootstrap_labels(val_feat)
test_feat = add_bootstrap_labels(test_feat)

CX_TARGETS = ["stress", "frustration", "urgency", "escalation_risk"]
print(train_feat[CX_TARGETS].describe())""")

# ---------------------------------------------------------------------------
# 10. Train XGBoost
# ---------------------------------------------------------------------------
md("## 10. Step 8 — Train XGBoost regressors (one per CX target)")
code("""from xgboost import XGBRegressor

FEATURE_COLS = [
    "emotion_angry", "emotion_happy", "emotion_neutral", "emotion_sad", "arousal", "valence",
    "pitch_mean", "pitch_std", "energy_mean", "energy_std",
    "speech_ratio", "speaking_rate", "pause_count", "pause_ratio",
]

X_train, X_val = train_feat[FEATURE_COLS], val_feat[FEATURE_COLS]

cx_models = {}
for target in CX_TARGETS:
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_train, train_feat[target], eval_set=[(X_val, val_feat[target])], verbose=False)
    cx_models[target] = model
    model.save_model(os.path.join(MODELS_DIR, f"{target}.json"))
    print(f"[TRAIN] {target}: best_iteration={model.best_iteration}, val_rmse={model.best_score:.4f}")

print("\\nAll 4 models trained and saved to", MODELS_DIR)""")

# ---------------------------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------------------------
md("""## 11. Step 9 — Evaluation on held-out test set (Session 5)

These numbers measure fit to the documented bootstrap formula (section 9
above) -- not real-world accuracy. Stated once, plainly, per
`CLAUDE_STAGE2.md`.""")
code("""import json
from sklearn.metrics import mean_absolute_error, r2_score

X_test = test_feat[FEATURE_COLS]
evaluation_results = {}
for target in CX_TARGETS:
    pred = cx_models[target].predict(X_test)
    true = test_feat[target].to_numpy()
    mae = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    r2 = r2_score(true, pred)
    evaluation_results[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{target:16s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

with open(os.path.join(RESULTS_DIR, "evaluation.json"), "w") as f:
    json.dump(evaluation_results, f, indent=2)

# Feature importance per target -- useful for the demo narrative.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, target in zip(axes.flat, CX_TARGETS):
    importances = cx_models[target].feature_importances_
    order = np.argsort(importances)[::-1]
    ax.barh([FEATURE_COLS[i] for i in order][:8][::-1], importances[order][:8][::-1], color="#4C72B0")
    ax.set_title(target)
fig.suptitle("Top feature importances per CX target (XGBoost)")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.show()
print("\\nSaved:", os.path.join(RESULTS_DIR, "evaluation.json"))
print("Saved:", os.path.join(RESULTS_DIR, "feature_importance.png"))""")

# ---------------------------------------------------------------------------
# 11b. Text-model-judged CX labels (optional upgrade over the bootstrap formula)
# ---------------------------------------------------------------------------
md("""## 11b. Step 9b — Text-model-judged CX labels (upgrade over the hand-written formula)

The formula in Step 7 is a fixed linear combination -- transparent, but
crude, and acoustic-only (it never looks at *what was said*, only how). This
section adds a second, free, fully-local signal: a pretrained fine-grained
text-emotion classifier (`SamLowe/roberta-base-go_emotions`, 28 GoEmotions
categories, from Hugging Face -- no API key, no cost, no external service)
reads the *actual IEMOCAP transcript* for every utterance, and a documented
mapping combines its emotion probabilities into
stress/frustration/urgency/escalation-risk. A second set of XGBoost models
is then trained on these text-derived labels.

**This is still not real ground truth** -- it's a second heuristic (a
pretrained classifier's emotion probabilities, mapped by a documented
formula) alongside the first one (direct acoustic signals), not measured
human ratings of real customer calls. Documented as such, same as the
formula-based labels in Step 7.

**No API key, no Kaggle Secret needed** -- this runs entirely on-device
(GPU if available, CPU otherwise), same as everything else in this
notebook. IEMOCAP is acted general dialogue, not real customer-service
audio -- another documented domain-mismatch limitation, same one Stage 1
already carries.""")

code("""!pip install -q transformers""")

code('''import glob as _glob

def find_iemocap_transcriptions(session_dirs):
    """Maps utt_id -> transcript text, parsing IEMOCAP's
    SessionX/dialog/transcriptions/*.txt files (format:
    "Ses01F_impro01_F000 [6.29-8.24]: Excuse me.")."""
    line_re = re.compile(r"^(\\S+)\\s*\\[[\\d.]+-[\\d.]+\\]:\\s*(.*)$")
    transcripts = {}
    for sdir in session_dirs:
        tx_dir = os.path.join(sdir, "dialog", "transcriptions")
        for txt_path in _glob.glob(os.path.join(tx_dir, "*.txt")):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = line_re.match(line.strip())
                    if not m:
                        continue
                    utt_id, text = m.groups()
                    if text.strip():
                        transcripts[utt_id] = text.strip()
    return transcripts

TRANSCRIPTS = find_iemocap_transcriptions(SESSION_DIRS)
print(f"[TEXT-LABEL] parsed {len(TRANSCRIPTS)} transcripts")

def attach_transcripts(feat_df):
    df = feat_df[feat_df["utt_id"].isin(TRANSCRIPTS)].reset_index(drop=True)
    df["transcript"] = df["utt_id"].map(TRANSCRIPTS)
    return df

text_train = attach_transcripts(train_feat)
text_val = attach_transcripts(val_feat)
text_test = attach_transcripts(test_feat)
print(f"[TEXT-LABEL] usable rows: train={len(text_train)} val={len(text_val)} test={len(text_test)}")''')

code('''from transformers import pipeline

# GoEmotions: 28 fine-grained categories (multi-label, sigmoid scores). Free,
# local, no API key. Runs on GPU automatically if available in this session.
emotion_clf = pipeline(
    "text-classification", model="SamLowe/roberta-base-go_emotions",
    top_k=None, device=0 if torch.cuda.is_available() else -1,
)
print("[TEXT-LABEL] loaded SamLowe/roberta-base-go_emotions")

def goemotions_batch(texts, batch_size=32):
    """Returns a list of {label: score} dicts, one per input text."""
    results = emotion_clf(list(texts), batch_size=batch_size, truncation=True)
    return [{d["label"]: d["score"] for d in row} for row in results]

_probe = goemotions_batch(["I have been on hold for twenty minutes, this is ridiculous."])
print("[VERIFY] sample scores (top 5):", sorted(_probe[0].items(), key=lambda kv: -kv[1])[:5])''')

code('''def add_text_model_labels(df):
    """Documented mapping from GoEmotions clusters -> the 4 CX targets.
    Same spirit as the acoustic formula in Step 7, different input signal
    (text content instead of audio prosody)."""
    scores = goemotions_batch(df["transcript"].tolist())
    g = pd.DataFrame(scores).fillna(0.0)

    anger_cluster = g.get("anger", 0) + g.get("annoyance", 0)
    fear_cluster = g.get("fear", 0) + g.get("nervousness", 0)
    sad_cluster = g.get("sadness", 0) + g.get("disappointment", 0) + g.get("grief", 0) + g.get("remorse", 0)
    disapproval_cluster = g.get("disgust", 0) + g.get("disapproval", 0)
    surprise = g.get("surprise", 0)
    neutral = g.get("neutral", 0)

    out = df.copy()
    out["stress"] = (0.35 * fear_cluster + 0.30 * anger_cluster + 0.20 * sad_cluster + 0.15 * surprise).clip(0, 1)
    out["frustration"] = (0.45 * anger_cluster + 0.25 * disapproval_cluster + 0.20 * sad_cluster + 0.10 * g.get("confusion", 0)).clip(0, 1)
    out["urgency"] = (0.40 * fear_cluster + 0.30 * anger_cluster + 0.20 * surprise + 0.10 * (1 - neutral)).clip(0, 1)
    out["escalation_risk"] = (0.5 * out["stress"] + 0.5 * out["frustration"]).clip(0, 1)
    return out

text_train_feat = add_text_model_labels(text_train)
text_val_feat = add_text_model_labels(text_val)
text_test_feat = add_text_model_labels(text_test)

for name, df in (("train", text_train_feat), ("val", text_val_feat), ("test", text_test_feat)):
    df.drop(columns=["transcript"]).to_csv(os.path.join(PROCESSED_DIR, f"{name}_text_labels.csv"), index=False)
print(f"[TEXT-LABEL] labeled train={len(text_train_feat)} val={len(text_val_feat)} test={len(text_test_feat)}")''')

code("""# Agreement between the two label sources on the same utterances -- does the
# acoustic-only formula roughly agree with the text-content-based one, or
# diverge? Divergence is expected and informative: acoustic prosody and
# spoken content don't always carry the same signal.
from scipy.stats import pearsonr as _pearsonr

_merged = test_feat.merge(text_test_feat, on="utt_id", suffixes=("_formula", "_text"))
print("Acoustic-formula vs. text-model label agreement (Pearson r, test split):")
for target in CX_TARGETS:
    r, _ = _pearsonr(_merged[f"{target}_formula"], _merged[f"{target}_text"])
    print(f"  {target:16s}: r={r:.3f}")""")

code("""import json
text_cx_models = {}
X_text_train = text_train_feat[FEATURE_COLS]
X_text_val = text_val_feat[FEATURE_COLS]
X_text_test = text_test_feat[FEATURE_COLS]

text_evaluation_results = {}
for target in CX_TARGETS:
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_text_train, text_train_feat[target], eval_set=[(X_text_val, text_val_feat[target])], verbose=False)
    text_cx_models[target] = model
    model.save_model(os.path.join(MODELS_DIR, f"{target}_textmodel.json"))

    pred = model.predict(X_text_test)
    true = text_test_feat[target].to_numpy()
    mae = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    r2 = r2_score(true, pred)
    text_evaluation_results[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{target:16s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  (n_train={len(X_text_train)})")

with open(os.path.join(RESULTS_DIR, "evaluation_textmodel.json"), "w") as f:
    json.dump(text_evaluation_results, f, indent=2)
print("\\nSaved:", os.path.join(RESULTS_DIR, "evaluation_textmodel.json"))
print("Saved models:", [os.path.join(MODELS_DIR, f'{t}_textmodel.json') for t in CX_TARGETS])""")

# ---------------------------------------------------------------------------
# 12. CXStateModel + real inference
# ---------------------------------------------------------------------------
md("""## 12. Step 10 — CXStateModel + real WAV inference

The stable interface for later AdaptiveCX integration:
`CXStateModel.predict(wav_path) -> {stress, frustration, urgency, escalation_risk}`""")
code("""class CXStateModel:
    def __init__(self, cx_models, feature_cols):
        self.cx_models = cx_models
        self.feature_cols = feature_cols

    def predict(self, wav_path):
        _t0 = time.time()
        feat = {}
        feat.update(stage1_predict(wav_path))
        feat.update(extract_acoustic_features(wav_path))
        x = pd.DataFrame([{c: feat[c] for c in self.feature_cols}])

        result = {target: float(model.predict(x)[0]) for target, model in self.cx_models.items()}
        result["latency_sec"] = time.time() - _t0
        return result

def print_cx_block(result):
    print("=" * 40)
    print("AdaptiveCX Stage 2 -- Customer State")
    print("=" * 40)
    print()
    for k in ("stress", "frustration", "urgency", "escalation_risk"):
        print(f"{k:16s}: {result[k]:.2f}")
    print()
    print(f"Inference latency: {result['latency_sec']:.3f} sec")
    print("=" * 40)

cx_model = CXStateModel(cx_models, FEATURE_COLS)
SAMPLE_WAV = test_df.iloc[0]["wav_path"]
result = cx_model.predict(SAMPLE_WAV)
print_cx_block(result)""")

# ---------------------------------------------------------------------------
# 13. Final report
# ---------------------------------------------------------------------------
md("## 13. Final report")
code("""import json
print("1. Project structure    :", BASE_DIR)
print("2. Stage 1 checkpoint   :", STAGE1_CKPT_PATH, "(reused, not retrained)")
print("3. Backbone             :", EMO2VEC_CHECKPOINT_ID, f"(via {EMO2VEC_SOURCE})")
print("4. Dataset              : IEMOCAP -", IEMOCAP_ROOT, "(same split as Stage 1)")
print("5. CX targets           :", CX_TARGETS, "(bootstrap/prototype labels -- NOT ground truth)")
print("6. Feature columns      :", FEATURE_COLS)
print("7. Evaluation results (acoustic-formula-labeled) :", json.dumps(evaluation_results, indent=2))
print("7b. Evaluation results (text-model-labeled)      :", json.dumps(text_evaluation_results, indent=2))
print("8. Saved models         :", [os.path.join(MODELS_DIR, f'{t}.json') for t in CX_TARGETS],
      "+ *_textmodel.json variants")
print("9. Real inference output printed above (Step 10)")
print("10. Known limitations   :")
print("    - Both label sources (the acoustic formula AND the GoEmotions-based text")
print("      mapping) are documented bootstrap/prototype scoring layers, not measured")
print("      customer state. See CLAUDE_STAGE2.md sections 2 and 9b for methodology.")
print("    - Interruptions/overlap features not computed (needs diarization).")
print("    - EmoWork (real stress/arousal/valence from call-center role-play) is")
print("      gated behind a Data Use Agreement -- documented as a future upgrade,")
print("      not used here.")
print("11. Ready for Phase 2   : CXStateModel(...).predict(wav) is the stable")
print("    interface for the later AdaptiveCX policy_engine.py integration. Pass")
print("    cx_models=text_cx_models to use the text-model-labeled variant instead of")
print("    the acoustic-formula-labeled default.")""")

# ---------------------------------------------------------------------------
# 14. Download
# ---------------------------------------------------------------------------
md("## 14. Download results")
code("""import shutil
from IPython.display import FileLink

DOWNLOAD_STAGE_DIR = "/kaggle/working/adaptivecx_stage2_download"
os.makedirs(DOWNLOAD_STAGE_DIR, exist_ok=True)
shutil.copytree(MODELS_DIR, os.path.join(DOWNLOAD_STAGE_DIR, "models"), dirs_exist_ok=True)
shutil.copytree(RESULTS_DIR, os.path.join(DOWNLOAD_STAGE_DIR, "results"), dirs_exist_ok=True)
for name in (
    "train_features.csv", "val_features.csv", "test_features.csv",
    "train_text_labels.csv", "val_text_labels.csv", "test_text_labels.csv",
):
    src = os.path.join(PROCESSED_DIR, name)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(DOWNLOAD_STAGE_DIR, name))

zip_base = "/kaggle/working/adaptivecx_stage2_results"
zip_path = shutil.make_archive(zip_base, "zip", DOWNLOAD_STAGE_DIR)
print("Zipped:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.2f} MB)")
FileLink(os.path.relpath(zip_path, "/kaggle/working"))""")

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("AdaptiveCX_Stage2_Kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote AdaptiveCX_Stage2_Kaggle.ipynb with {len(cells)} cells")
