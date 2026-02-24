import os
import librosa
import numpy as np
import cv2

# =============================
# CONFIG
# =============================
AUDIO_DIR = "testing/audio"
OUTPUT_DIR = "testing/spectogram_testing"

SR = 32000
DURATION = 5
SAMPLES_PER_CHUNK = SR * DURATION

os.makedirs(OUTPUT_DIR, exist_ok=True)


def audio_to_stft_image(audio_chunk):
    D = librosa.stft(
        audio_chunk,
        n_fft=1024,
        hop_length=512
    )

    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    S_db -= S_db.min()
    S_db /= (S_db.max() + 1e-6)
    S_db *= 255.0

    S_db = S_db.astype(np.uint8)

    S_resized = cv2.resize(S_db, (512, 512))

    return S_resized


def process_audio_file(audio_path):
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    y, _ = librosa.load(audio_path, sr=SR)

    chunk_index = 0

    for start in range(0, len(y), SAMPLES_PER_CHUNK):
        end = start + SAMPLES_PER_CHUNK
        chunk = y[start:end]

        if len(chunk) < SAMPLES_PER_CHUNK:
            chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))

        # skip silent chunks
        if np.mean(np.abs(chunk)) < 0.01:
            continue

        spec_img = audio_to_stft_image(chunk)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{filename}_{chunk_index}.png"
        )

        cv2.imwrite(output_path, spec_img)

        chunk_index += 1


# =============================
# MAIN
# =============================
for file in os.listdir(AUDIO_DIR):
    if file.lower().endswith((".wav", ".mp3", ".ogg")):
        full_path = os.path.join(AUDIO_DIR, file)
        print(f"Processing {file}")
        process_audio_file(full_path)

print("Done.")