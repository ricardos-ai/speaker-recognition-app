# streamlit_app.py
import streamlit as st
import torch
import joblib
from speechbrain.inference import SpeakerRecognition
import torchaudio
import torch.nn as nn
from pydub import AudioSegment
import numpy as np
import os

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label encoder
label_encoder = joblib.load("speaker_label_encoder.pkl")
num_classes = len(label_encoder.classes_)

# Classifier model
class SimpleEmbeddingClassifierBN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
model = SimpleEmbeddingClassifierBN(192, num_classes)
model.load_state_dict(torch.load("best_speaker_classifier.pt", map_location=device))
model.eval()

# Load ECAPA-TDNN
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
verification.eval()

# Preprocess function
def preprocess_audio(filepath, target_sr=16000, max_duration_sec=12):
    torchaudio.set_audio_backend("sox_io")

    audio = AudioSegment.from_file(filepath)
    samples = np.array(audio.get_array_of_samples())

    # Detect sample width to choose proper dtype
    sample_width = audio.sample_width
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("Unsupported sample width: {}".format(sample_width))

    samples = samples.astype(dtype)
    samples = samples / np.iinfo(dtype).max  # Normalize to [-1, 1]
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    sample_rate = audio.frame_rate

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    # Pad/truncate
    max_samples = target_sr * max_duration_sec
    if waveform.shape[1] < max_samples:
        pad = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :max_samples]

    return waveform

# === Streamlit UI ===
st.title("🎙️ Speaker Recognition App")
st.write("Upload an audio file (.mp3, .ogg, .wav) to identify the speaker.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "ogg", "wav"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    waveform = preprocess_audio("temp_audio.mp3")

    with torch.no_grad():
        emb = verification.encode_batch(waveform, wav_lens=torch.tensor([1.0]))
        emb = emb.squeeze(0).squeeze(0)
        emb = torch.nn.functional.normalize(emb, p=2, dim=0).to(device)

        output = model(emb.unsqueeze(0))
        pred = torch.argmax(output, dim=1).item()
        speaker = label_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Predicted speaker: **{speaker}**")
