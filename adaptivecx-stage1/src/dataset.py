"""IEMOCAP metadata loading, label mapping, speaker-independent splitting, and
cached embedding extraction (Sections 4-7 of the project spec).

Never invent labels: emotion/arousal/valence all come straight from IEMOCAP's
EmoEvaluation annotations, only rescaled/renamed, never fabricated.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import audio as audio_mod

# ang -> angry, hap+exc -> happy, sad -> sad, neu -> neutral.
# All other raw labels (fru, sur, fea, dis, oth, xxx, ...) are dropped: they
# are either low-frequency, ambiguous, or non-emotion categories in IEMOCAP
# and are not part of the target label set (project spec Section 4).
EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
}
EMOTION_CLASSES = sorted(set(EMOTION_MAP.values()))  # ['angry', 'happy', 'neutral', 'sad']
EMOTION_TO_IDX = {c: i for i, c in enumerate(EMOTION_CLASSES)}

_LINE_RE = re.compile(
    r"^\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)\s+\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"
)


def find_iemocap_root(search_root):
    """Locates an attached IEMOCAP dataset by its distinctive EmoEvaluation dirs."""
    eval_dirs = glob.glob(os.path.join(search_root, "**", "dialog", "EmoEvaluation"), recursive=True)
    if not eval_dirs:
        raise FileNotFoundError(
            f"Could not find an 'EmoEvaluation' directory under {search_root}. "
            "Point --data-root at a local IEMOCAP copy (SessionX/dialog/EmoEvaluation/*.txt "
            "+ SessionX/sentences/wav/**/*.wav layout)."
        )
    session_dirs = sorted({os.path.dirname(os.path.dirname(p)) for p in eval_dirs})
    dataset_roots = sorted({os.path.dirname(s) for s in session_dirs})
    return dataset_roots[0], session_dirs


def parse_session(session_dir):
    rows = []
    eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
    wav_root = os.path.join(session_dir, "sentences", "wav")
    for txt_path in glob.glob(os.path.join(eval_dir, "*.txt")):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _LINE_RE.match(line.strip())
                if not m:
                    continue
                _, _, utt_id, emo_raw, val, act, dom = m.groups()
                dialog_id = utt_id.rsplit("_", 1)[0]
                wav_path = os.path.join(wav_root, dialog_id, utt_id + ".wav")
                session_num = utt_id[3:5]
                speaker_tag = utt_id.split("_")[-1][0]  # 'F' or 'M'
                rows.append({
                    "utt_id": utt_id,
                    "wav_path": wav_path,
                    "session": int(session_num),
                    "speaker_id": f"Ses{session_num}_{speaker_tag}",
                    "emotion_raw": emo_raw,
                    "valence_raw": float(val),
                    "arousal_raw": float(act),
                    "dominance_raw": float(dom),
                })
    return rows


def build_metadata(session_dirs):
    """Parses all sessions, maps to the 4-class label set, normalizes arousal/valence to [-1, 1]."""
    all_rows = []
    for sdir in session_dirs:
        all_rows.extend(parse_session(sdir))
    meta = pd.DataFrame(all_rows)
    print("[DATA] parsed utterances (pre wav-check):", len(meta))

    meta["wav_exists"] = meta["wav_path"].apply(os.path.isfile)
    n_missing = (~meta["wav_exists"]).sum()
    if n_missing:
        print(f"[DATA] dropping {n_missing} rows with no matching wav file")
    meta = meta[meta["wav_exists"]].drop(columns=["wav_exists"]).reset_index(drop=True)

    meta["emotion"] = meta["emotion_raw"].map(EMOTION_MAP)
    n_before = len(meta)
    meta = meta.dropna(subset=["emotion"]).reset_index(drop=True)
    print(f"[DATA] kept {len(meta)} / {n_before} utterances after 4-class emotion mapping")

    # IEMOCAP's empirically-confirmed 1-5 SAM scale -> [-1, 1]
    meta["valence"] = (meta["valence_raw"] - 3.0) / 2.0
    meta["arousal"] = (meta["arousal_raw"] - 3.0) / 2.0
    meta["emotion_idx"] = meta["emotion"].map(EMOTION_TO_IDX)
    return meta


def speaker_independent_split(meta):
    """Sessions 1-3 = train, 4 = val, 5 = test. IEMOCAP has 10 speakers, 2 per
    session, disjoint across sessions -> zero speaker leakage by construction."""
    train_df = meta[meta["session"].isin([1, 2, 3])].reset_index(drop=True)
    val_df = meta[meta["session"] == 4].reset_index(drop=True)
    test_df = meta[meta["session"] == 5].reset_index(drop=True)

    train_speakers = set(train_df["speaker_id"])
    val_speakers = set(val_df["speaker_id"])
    test_speakers = set(test_df["speaker_id"])
    overlap = (train_speakers & val_speakers) | (train_speakers & test_speakers) | (val_speakers & test_speakers)
    assert not overlap, f"Speaker leakage detected across splits: {overlap}"
    return train_df, val_df, test_df


def extract_split_embeddings(df, split_name, embed_fn, embed_dim, cache_dir, funasr_output_dir):
    """Runs embed_fn(wav_path) per row, caching to cache_dir/<utt_id>.npy. Drops
    and reports any utterance that fails to load or embed."""
    os.makedirs(cache_dir, exist_ok=True)
    embeddings = np.zeros((len(df), embed_dim), dtype=np.float32)
    keep_mask = np.ones(len(df), dtype=bool)
    failures = []

    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"extract[{split_name}]")):
        cache_path = os.path.join(cache_dir, f"{row.utt_id}.npy")
        try:
            if os.path.isfile(cache_path):
                emb = np.load(cache_path)
            else:
                audio_mod.load_and_preprocess_audio(row.wav_path)  # validation pass
                emb = embed_fn(row.wav_path, funasr_output_dir)
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


class EmbeddingDataset(torch.utils.data.Dataset):
    """Wraps a precomputed [N, embed_dim] embedding matrix + its label columns."""

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
