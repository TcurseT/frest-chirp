"""
Bird Species Prediction API
Pipeline: Audio upload → torchaudio.Spectrogram (n_fft=400, hop=200, power=2) → AmplitudeToDB
         → normalise [0,255] → 512×512 grayscale → RGB → EfficientNet-B0 inference

Spectrogram parameters MUST match audio_to_spectogram.py (training data creator).
"""

import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
import torchaudio.transforms as TAT
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision.models import efficientnet_b0
from torchvision import transforms

# ─────────────────────────── Config ───────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR       = os.path.join(BASE_DIR, "model")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

SR               = 32000
DURATION         = 5
SAMPLES_PER_CHUNK = SR * DURATION
SILENCE_THRESHOLD = 0.001  # lowered: 0.01 was too aggressive for quiet/distant bird calls

# ─────────────────────────── Load class names ─────────────────
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
print(f"[API] Loaded {NUM_CLASSES} classes from {CLASS_NAMES_PATH}")

# ─────────────────────────── Image transform ──────────────────
# Must EXACTLY match the transform used in training/train.py
# Training images are grayscale L-mode PNGs; ImageFolder calls .convert('RGB')
# so we replicate that same behaviour: load as L, convert to RGB, apply transform.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────── Model cache ──────────────────────
_model_cache: dict = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[API] Using device: {DEVICE}")


def get_model(model_name: str):
    """Load (and cache) the requested .pth checkpoint."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[API] Loading model: {model_path}")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    _model_cache[model_name] = model
    print(f"[API] Model loaded and cached: {model_name}")
    return model


# ─────────────────────────── Audio helpers ────────────────────

def chunk_to_stft_tensor(chunk: np.ndarray) -> torch.Tensor:
    """
    Convert a single 5-second mono audio chunk → normalised tensor (1×3×224×224).

    Pipeline EXACTLY matches audio_to_spectogram.py (which generated the training images):
      1. Convert numpy chunk → torch tensor [1, samples]
      2. torchaudio.Spectrogram(n_fft=400, hop_length=200, power=2)  ← DEFAULT params
      3. torchaudio.AmplitudeToDB()                                   ← same as training
      4. Normalise to [0, 255] uint8  (per-chunk min-max, same as training)
      5. Resize to 512×512 grayscale with cv2
      6. PIL: L → RGB  (replicates ImageFolder .convert('RGB'))
      7. Resize(224,224) → ToTensor → Normalize
    """
    # 1: numpy → torch [1, samples]
    waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)

    # 2-3: Power spectrogram → dB  (identical to audio_to_spectogram.py)
    spec    = TAT.Spectrogram(n_fft=400, hop_length=200, power=2.0)(waveform)  # [1, F, T]
    spec_db = TAT.AmplitudeToDB()(spec)                                          # [1, F, T]

    # 4: [0, 255] uint8 normalisation
    S = spec_db.squeeze().numpy()     # [F, T]
    S = S - S.min()
    S = S / (S.max() + 1e-8) * 255.0
    S = S.astype(np.uint8)

    # 5: resize to 512×512
    S_resized = cv2.resize(S, (512, 512))

    # 6: PIL L → RGB  (replicates ImageFolder .convert('RGB') at training time)
    pil_L   = Image.fromarray(S_resized, mode="L")
    pil_rgb = pil_L.convert("RGB")

    # 7: apply training transform
    tensor = TRANSFORM(pil_rgb).unsqueeze(0).to(DEVICE)   # 1×3×224×224
    return tensor


def audio_bytes_to_tensors(audio_bytes: bytes) -> list:
    """
    Load audio, split into 5-second chunks, discard silent chunks,
    and return a list of tensors — one per valid chunk.

    This mirrors the full audio_to_spectogram.py pipeline exactly.
    """
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR)
    print(f"[DIAG] Audio loaded: duration={len(y)/SR:.2f}s | RMS={np.sqrt(np.mean(y**2)):.5f} | samples={len(y)}")

    tensors = []
    for start in range(0, len(y), SAMPLES_PER_CHUNK):
        chunk = y[start : start + SAMPLES_PER_CHUNK]

        # Pad last chunk if it's shorter than 5 seconds
        if len(chunk) < SAMPLES_PER_CHUNK:
            chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))

        # Skip silent chunks (identical threshold to audio_to_spectogram.py)
        if np.mean(np.abs(chunk)) < SILENCE_THRESHOLD:
            continue

        tensors.append(chunk_to_stft_tensor(chunk))

    return tensors


# ─────────────────────────── Flask app ────────────────────────
app = Flask(__name__)
CORS(app)   # allow the Vite dev server (localhost:5173) to call us


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": NUM_CLASSES, "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    # ── 1. Parse request ──────────────────────────────────────
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided (field name: 'audio')"}), 400

    audio_file  = request.files["audio"]
    model_name  = request.form.get("model", "bird_model_epoch8.pth")
    audio_bytes = audio_file.read()

    # Only allow checkpoint files that exist in /model
    allowed = {"bird_model_epoch5.pth", "bird_model_epoch8.pth", "bird_model.pth"}
    if model_name not in allowed:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # ── 2. Split audio → per-chunk tensors (skip silence) ─────
    try:
        tensors = audio_bytes_to_tensors(audio_bytes)
    except Exception as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 422

    if not tensors:
        return jsonify({"error": "No non-silent audio found in the recording. "
                                 "Please ensure the audio contains bird calls."}), 422

    # ── 3. Load model ─────────────────────────────────────────
    try:
        model = get_model(model_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # ── 4. Inference: average softmax across ALL valid chunks ──
    # Averaging probabilities (not logits) gives a proper ensemble.
    avg_probs = torch.zeros(NUM_CLASSES, device=DEVICE)

    with torch.no_grad():
        for tensor in tensors:
            logits = model(tensor)                 # 1×N
            probs  = F.softmax(logits, dim=1)[0]   # N
            avg_probs += probs

    avg_probs /= len(tensors)                      # normalise to sum=1

    top5_probs, top5_idx = torch.topk(avg_probs, 5)
    top1_idx  = top5_idx[0].item()
    top1_prob = top5_probs[0].item()

    print(f"[API] {model_name} | chunks={len(tensors)} | "
          f"pred={CLASS_NAMES[top1_idx]} | conf={top1_prob*100:.1f}%")

    # ── 5. Build response ─────────────────────────────────────
    top5 = [
        {"name": CLASS_NAMES[i.item()], "confidence": round(p.item() * 100, 2)}
        for i, p in zip(top5_idx, top5_probs)
    ]

    return jsonify({
        "prediction": CLASS_NAMES[top1_idx],
        "confidence": round(top1_prob * 100, 2),
        "top5":       top5,
        "model":      model_name,
    })


if __name__ == "__main__":
    # Pre-load both models on startup so first request is instant
    for m in ("bird_model_epoch5.pth", "bird_model_epoch8.pth"):
        try:
            get_model(m)
        except FileNotFoundError:
            print(f"[API] Warning: {m} not found, skipped pre-load")

    app.run(host="0.0.0.0", port=5000, debug=False)
