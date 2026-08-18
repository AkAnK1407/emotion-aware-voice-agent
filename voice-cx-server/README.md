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

## Redeploying fast after a sandbox wipe (skip the slow model download)

If your EC2 instance gets torn down by a time-limited sandbox and you need
to redeploy from scratch, the single slowest, most variable step is
`funasr` downloading the ~1.1GB `emotion2vec+` checkpoint from ModelScope
(observed anywhere from 70s to 18 minutes depending on network conditions
that day). Skip it entirely by uploading a previously-downloaded copy
instead of letting the new instance re-download it:

```powershell
# On your local machine, IF you've run this before, the full checkpoint is
# already cached at:
#   C:\Users\<you>\.cache\modelscope\models\iic--emotion2vec_plus_base\snapshots\master\
# Upload that whole folder to the new instance BEFORE starting the service:
scp -i keypairN.pem -r `
    "C:\Users\<you>\.cache\modelscope\models\iic--emotion2vec_plus_base" `
    ubuntu@<new-ec2-host>:/home/ubuntu/.cache/modelscope/models/
```

Then start the service as usual — `funasr`/`modelscope` check this cache
path before hitting the network, so it loads directly from the uploaded
file (roughly a minute, mostly just deserializing the checkpoint into
memory) instead of downloading anything.

If you don't have a local cache yet, run the deploy once normally (letting
it download), and it'll be cached locally afterward via `main.py`'s own
download step -- Actually it downloads *on the server*, not locally. To get
a local copy, either run `adaptivecx-stage1/scripts/predict.py` once
locally (it downloads the same checkpoint to the same local cache path via
the same `funasr`/`modelscope` mechanism), or `scp` it back down from a
working server instance:
```powershell
scp -i keypairN.pem -r `
    ubuntu@<working-ec2-host>:/home/ubuntu/.cache/modelscope/models/iic--emotion2vec_plus_base `
    "C:\Users\<you>\.cache\modelscope\models\"
```

## Notes

- First request after startup is slow (backbone still loading in the
  background) — `/health` reports `model_loaded: false` until ready.
  Afterward, predictions take roughly as long as the audio clip itself.
- Models in `models/` are copied from `adaptivecx-stage1/models/best_stage1.pt`
  and `adaptivecx-stage2/models/{target}.json` in the main project repo — if
  those get retrained, re-copy the updated files here and restart the service.
