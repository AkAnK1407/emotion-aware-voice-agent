# AdaptiveCX Voice CX Server

Standalone FastAPI service wrapping Stage 1 (emotion2vec+) + Stage 2 Version A
(acoustic-formula XGBoost). Exists because the live agent's local machine
doesn't have enough RAM to run `torch`+`funasr` reliably alongside everything
else — this runs on a separate, properly-sized server instead, and the agent
calls it over HTTP per turn.

Shadow mode: this only feeds the dashboard's experimental "Voice-Based CX"
panel. It does not drive the agent's actual response — that logic stays on
the existing, faster, already-proven text-based `emotion_engine.py` /
`policy_engine.py`.

## Deploy (Ubuntu 22.04, e.g. an EC2 t3.small/t3.medium)

```bash
# On the server, as the ubuntu user:
sudo apt update && sudo apt install -y python3-venv python3-pip

cd ~/voice-cx-server
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Test it runs (first request will trigger the ~1-4 min model load in the background)
.venv/bin/python main.py
# In another terminal: curl http://localhost:8000/health

# Run it as a persistent service (auto-restarts, survives reboots):
sudo cp adaptivecx-voice-cx.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now adaptivecx-voice-cx
sudo systemctl status adaptivecx-voice-cx
```

## API

- `GET /health` → `{"status": "ok", "model_loaded": true|false}`
- `POST /predict` (multipart form, field name `audio`, a WAV file) →
  `{"emotion": "...", "arousal": ..., "valence": ..., "stress": ..., "frustration": ..., "urgency": ..., "escalation_risk": ...}`

Test from your local machine once the security group allows port 8000 from your IP:
```bash
curl -X POST -F "audio=@samples/test.wav" http://<EC2_PUBLIC_IP>:8000/predict
```

## Notes

- First request after startup is slow (backbone still loading in the
  background) — `/health` reports `model_loaded: false` until ready.
  Afterward, predictions take roughly as long as the audio clip itself.
- Models in `models/` are copied from `adaptivecx-stage1/models/best_stage1.pt`
  and `adaptivecx-stage2/models/{target}.json` in the main project repo — if
  those get retrained, re-copy the updated files here and restart the service.
