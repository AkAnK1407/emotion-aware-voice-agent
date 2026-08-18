"""
main.py — AdaptiveCX Voice-CX Server

Small FastAPI service wrapping voice_cx_model.py (Stage 1 + Stage 2
Version A). Loads the model once at startup and serves predictions over
HTTP so the live agent (running on a low-RAM local machine) doesn't have
to hold torch/funasr in its own process.

Run directly for local testing:
    python main.py

Run in production (see systemd unit adaptivecx-voice-cx.service):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import voice_cx_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-cx-server")

app = FastAPI(title="AdaptiveCX Voice CX Server")

_model = voice_cx_model.get_instance()

# Strictly one inference at a time -- this box has just enough RAM for one
# model forward pass. Letting concurrent requests each spawn their own
# executor-thread inference was spiking memory enough to crash the process
# (observed: repeated 503s / connection resets under rapid-fire requests).
# A caller hitting this while busy gets an immediate 503, not a queued wait
# -- callers (the live agent) are expected to skip rather than retry, since
# this is a display-only shadow-mode feature.
_busy = False


@app.on_event("startup")
async def _startup():
    """Loads the model once, in a thread, so startup doesn't block the
    event loop -- but requests made before it finishes will 503 until it's
    ready (see /health)."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _model.load)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model._loaded}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """Accepts a WAV file upload, returns
    {emotion, arousal, valence, stress, frustration, urgency, escalation_risk}."""
    global _busy

    if not _model._loaded:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    if _busy:
        raise HTTPException(status_code=503, detail="Busy with another request, skip this one")

    _busy = True
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _model.predict_from_wav_path, tmp_path)
        return JSONResponse(result)
    except Exception as e:
        logger.warning(f"[predict] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _busy = False
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
