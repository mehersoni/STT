# Real-Time STT UI with Your CNN+BiLSTM Mel Spectrogram Model

import sounddevice as sd
import torch
import torchaudio
import numpy as np
import tkinter as tk
from threading import Thread
from torchaudio.transforms import MelSpectrogram, Resample
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# === Load Pretrained Model and Vocab ===
from cnn_lstm_model import CNNBiLSTMModel, CHAR_TO_INDEX, INDEX_TO_CHAR  # Assumes you've saved your model + vocab here

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_epoch15.pt"
SAMPLE_RATE = 16000

# === Decoder ===
def greedy_decoder(output):
    output = torch.argmax(output, dim=-1)
    results = []
    for sequence in output:
        decoded = []
        prev = 0
        for token in sequence:
            if token != prev and token != 0:
                decoded.append(INDEX_TO_CHAR[token.item()])
            prev = token
        results.append(''.join(decoded))
    return results

# === Load Model ===
model = CNNBiLSTMModel(input_dim=80, hidden_dim=256, vocab_size=len(CHAR_TO_INDEX))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Audio Preprocessing ===
mel_extractor = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)
resampler = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

recording = False
recorded_audio = None

# === UI Callbacks ===
def start_recording():
    global recording, recorded_audio
    recorded_audio = None
    recording = True
    status_label.config(text="Recording...")
    Thread(target=record_loop).start()

def stop_recording():
    global recording
    recording = False
    status_label.config(text="Transcribing...")
    Thread(target=run_transcription).start()

def record_loop():
    global recorded_audio
    audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recorded_audio = torch.tensor(audio).squeeze().float()
    recorded_audio = recorded_audio / (recorded_audio.abs().max() + 1e-5)  # Normalize audio
    print("[DEBUG] Audio captured, samples:", len(recorded_audio))
    print("[DEBUG] Max value:", recorded_audio.max().item(), "Min value:", recorded_audio.min().item())
    status_label.config(text="Recording stopped. Click 'Stop' to transcribe.")

def transcribe(audio):
    if audio.shape[0] != SAMPLE_RATE * 5:
        audio = torch.nn.functional.pad(audio, (0, SAMPLE_RATE * 5 - audio.shape[0]))

    audio = resampler(audio)
    mel = mel_extractor(audio).clamp(min=1e-5)  # Remove log2, use clean mel directly
    print("[DEBUG] Mel shape:", mel.shape)

    # Show mel spectrogram for debug
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.log10().numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.title("Mel Spectrogram (log10 scale)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    mel = mel.unsqueeze(0).transpose(1, 2).to(DEVICE)  # [1, time, n_mels]

    with torch.no_grad():
        output = model(mel)  # [1, time, vocab]
        print("[DEBUG] Raw token predictions:", torch.argmax(output, dim=-1))
        log_probs = output.log_softmax(2)
        decoded = greedy_decoder(log_probs)[0]
        print("[DEBUG] Decoded output:", decoded)

    return decoded

def run_transcription():
    global recorded_audio
    if recorded_audio is not None and recorded_audio.numel() > 0:
        result = transcribe(recorded_audio)
        output_text.set(result)
        status_label.config(text="Done.")
    else:
        output_text.set("No audio recorded.")
        status_label.config(text="Idle")

# === Tkinter UI ===
root = tk.Tk()
root.title("Real-Time STT (Mel Spectrogram Model)")
root.geometry("650x300")

output_text = tk.StringVar()
status_label = tk.Label(root, text="Idle", font=("Arial", 12), fg="blue")
status_label.pack(pady=5)

tk.Label(root, text="Transcribed Text:", font=("Arial", 14)).pack(pady=5)
tk.Label(root, textvariable=output_text, wraplength=600, font=("Arial", 12)).pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Start Recording", command=start_recording, font=("Arial", 12), width=20)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop & Transcribe", command=stop_recording, font=("Arial", 12), width=20)
stop_btn.grid(row=0, column=1, padx=10)

root.mainloop()