# STT


This document outlines all required Python files and steps for building a complete Speech-to-Text (STT) pipeline using the LibriSpeech dataset, Mel Spectrograms, and a CNN + BiLSTM model.


File Structure

project_root/
├── data/
│   └── LibriSpeech/                      # Dataset (train-clean-100, dev-clean, test-clean)
├── melspec_data/
│   ├── train/                            # Preprocessed mel spectrograms
│   ├── dev/
│   └── test/
├── preprocess_librispeech.py            # Convert audio to mel and save .pt
├── cnn_lstm_model.py                    # Model definition
├── train_stt_model.ipynb                # Model training code
├── real_time_stt_ui.py                  # Real-time UI using trained model
├── README.md                            # Project instructions
└── model_epochX.pt                      # Trained model checkpoints
