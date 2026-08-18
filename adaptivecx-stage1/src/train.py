"""Stage A (frozen-backbone) training loop with early stopping (Sections 11-13)."""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error

from .model import EmotionHeads


def multitask_loss(emotion_logits, arousal_pred, valence_pred,
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
    return total, {"emotion": l_emotion.item(), "arousal": l_arousal.item(), "valence": l_valence.item()}


def run_epoch(heads, loader, optimizer, loss_weights, device, train_mode):
    heads.train(train_mode)
    total_loss = 0.0
    all_emotion_true, all_emotion_pred = [], []
    all_arousal_true, all_arousal_pred = [], []
    all_valence_true, all_valence_pred = [], []

    for emb, emo, aro, val in loader:
        emb, emo, aro, val = emb.to(device), emo.to(device), aro.to(device), val.to(device)
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

    return {
        "loss": total_loss / len(loader.dataset),
        "emotion_f1": f1_score(all_emotion_true, all_emotion_pred, average="macro", zero_division=0),
        "arousal_mae": mean_absolute_error(all_arousal_true, all_arousal_pred),
        "valence_mae": mean_absolute_error(all_valence_true, all_valence_pred),
    }


def train_stage_a(train_ds, val_ds, embed_dim, emotion_classes, config, models_dir, device,
                   checkpoint_name="best_stage1.pt"):
    """Trains EmotionHeads on top of frozen-backbone embeddings. Saves the best
    checkpoint (by val loss) to models_dir/checkpoint_name and returns its path."""
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    heads = EmotionHeads(input_dim=embed_dim, num_emotions=len(emotion_classes)).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=config["training"]["learning_rate"])
    loss_weights = {
        "emotion": config["loss"]["emotion_weight"],
        "arousal": config["loss"]["arousal_weight"],
        "valence": config["loss"]["valence_weight"],
    }

    os.makedirs(models_dir, exist_ok=True)
    best_ckpt_path = os.path.join(models_dir, checkpoint_name)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = run_epoch(heads, train_loader, optimizer, loss_weights, device, train_mode=True)
        val_metrics = run_epoch(heads, val_loader, optimizer, loss_weights, device, train_mode=False)

        print(f"Epoch {epoch}/{config['training']['epochs']}")
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
                "embed_dim": embed_dim,
                "emotion_classes": emotion_classes,
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["training"]["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['training']['patience']} epochs)")
                break
        print()

    print("Training complete. Best checkpoint:", best_ckpt_path)
    return best_ckpt_path
