# 🎙Speech-to-Text (STT) Pipeline using LibriSpeech, Mel Spectrograms, and CNN + BiLSTM

This repository contains a complete pipeline for building a Speech-to-Text (STT) system using the LibriSpeech dataset(https://www.openslr.org/12), Mel Spectrograms, and a deep learning model combining Convolutional Neural Networks (CNN) with Bidirectional LSTMs (BiLSTM).

---

## Project Structure

```
project_root/
├── data/
│   └── LibriSpeech/              # Original dataset (train-clean-100, dev-clean, test-clean)
├── melspec_data/
│   ├── train/                    # Preprocessed mel spectrograms
│   ├── dev/
│   └── test/
├── preprocess_librispeech.py    # Converts raw audio to mel spectrograms (.pt)
├── cnn_lstm_model.py            # CNN + BiLSTM model architecture
├── train_stt_model.ipynb        # Model training notebook
├── real_time_stt_ui.py          # Real-time speech-to-text UI using trained model
├── model_epochX.pt              # Trained model checkpoints
└── README.md                    # Project instructions
```

---

## Model Architecture

* **Feature Extraction:** Mel Spectrogram (torchaudio)
* **Encoder:** 2D CNN layers to capture local acoustic features
* **Sequence Model:** Bidirectional LSTM for temporal context
* **Output Layer:** Linear + CTC Loss for sequence alignment

---

## Pipeline Overview

1. **Download LibriSpeech Dataset:**

   * Use `train-clean-100`, `dev-clean`, and `test-clean`.

2. **Preprocess Audio:**

   ```bash
   python preprocess_librispeech.py
   ```

   * Converts `.flac` audio to mel spectrograms and saves `.pt` files in `melspec_data/`.

3. **Train the Model:**

   * Open and run `train_stt_model.ipynb` in Jupyter/Colab.
   * Supports multi-GPU training via PyTorch Lightning or manual control.

4. **Evaluate:**

   * Evaluate Word Error Rate (WER) on dev/test sets.

5. **Real-Time Inference:**

   ```bash
   python real_time_stt_ui.py
   ```

   * Loads trained model and performs real-time transcription using microphone input.

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchaudio
* numpy
* librosa
* matplotlib
* tqdm
* PyQt5 (for real-time UI)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Sample Results

| Metric  | dev-clean  |    test-clean        |
|         |            |                      |
| WER (%) | 0.4476     |        0.4476        |
| CER (%) | 0.1598     |        0.1598        |



---

## To Do

* [x] Preprocessing script
* [x] CNN + BiLSTM model
* [x] Training pipeline
* [x] Real-time UI
* [ ] Beam Search Decoder
* [ ] Attention mechanism integration

---

## References

* [LibriSpeech ASR corpus](https://www.openslr.org/12)
* [CTC Loss paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
* [DeepSpeech2](https://arxiv.org/abs/1512.02595)

