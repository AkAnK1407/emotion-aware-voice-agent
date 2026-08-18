"""Held-out test set evaluation + confusion matrix plot (Section 14)."""
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from .model import EmotionHeads


def evaluate_checkpoint(checkpoint_path, test_ds, device, results_dir, batch_size=8):
    ckpt = torch.load(checkpoint_path, map_location=device)
    heads = EmotionHeads(input_dim=ckpt["embed_dim"], num_emotions=len(ckpt["emotion_classes"])).to(device)
    heads.load_state_dict(ckpt["model_state_dict"])
    heads.eval()
    emotion_classes = ckpt["emotion_classes"]

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    emo_true, emo_pred = [], []
    aro_true, aro_pred = [], []
    val_true, val_pred = [], []

    with torch.no_grad():
        for emb, emo, aro, val in test_loader:
            emb = emb.to(device)
            emo_logits, aro_p, val_p = heads(emb)
            emo_true.extend(emo.numpy().tolist())
            emo_pred.extend(emo_logits.argmax(dim=-1).cpu().numpy().tolist())
            aro_true.extend(aro.numpy().tolist())
            aro_pred.extend(aro_p.cpu().numpy().tolist())
            val_true.extend(val.numpy().tolist())
            val_pred.extend(val_p.cpu().numpy().tolist())

    accuracy = accuracy_score(emo_true, emo_pred)
    macro_f1 = f1_score(emo_true, emo_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(emo_true, emo_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(emo_true, emo_pred, labels=list(range(len(emotion_classes))))

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
                    "confusion_matrix": cm.tolist(), "classes": emotion_classes},
        "arousal": {"mae": aro_mae, "rmse": aro_rmse, "pearson": aro_pearson, "spearman": aro_spearman},
        "valence": {"mae": val_mae, "rmse": val_rmse, "pearson": val_pearson, "spearman": val_spearman},
        "test_set_size": len(test_ds),
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    _save_confusion_matrix(cm, emotion_classes, os.path.join(results_dir, "confusion_matrix.png"))

    print("\nSaved:", os.path.join(results_dir, "evaluation.json"))
    print("Saved:", os.path.join(results_dir, "confusion_matrix.png"))
    return evaluation_results


def _save_confusion_matrix(cm, classes, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
