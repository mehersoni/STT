import torchaudio
import os
import csv
from tqdm import tqdm

# === CONFIGURATION ===
LIBRISPEECH_DIR = "C:/DRDO Project/LibriSpeech"             # Original dataset path
MELSPEC_DIR = "C:/DRDO Project/LibriSpeech_MELSPEC"         # Mel spectrogram path
OUTPUT_METADATA_DIR = "C:/DRDO Project/LibriSpeech_Metadata" # Output CSV path
os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)

# === Function to create metadata CSV for a split ===
def create_metadata(split_url, split_name):
    print(f"\n⏳ Processing split: {split_name}")
    dataset = torchaudio.datasets.LIBRISPEECH(root=LIBRISPEECH_DIR, url=split_url, download=False)

    output_csv_path = os.path.join(OUTPUT_METADATA_DIR, f"{split_name}.csv")
    mel_split_folder = os.path.join(MELSPEC_DIR, split_name)

    missing = 0

    with open(output_csv_path, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "transcription"])

        for i in tqdm(range(len(dataset)), desc=f"Aligning {split_name}", unit="samples"):
            waveform, sr, transcript, speaker_id, chapter_id, utterance_id = dataset[i]

            # Construct expected mel spectrogram file path
            filename = f"{speaker_id}-{chapter_id}-{utterance_id}.pt"
            mel_path = os.path.join(mel_split_folder, filename)

            if os.path.exists(mel_path):
                writer.writerow([mel_path, transcript])
            else:
                missing += 1

    print(f"✅ Done. Saved: {output_csv_path}")
    if missing > 0:
        print(f"⚠️ {missing} MelSpectrogram files were missing in {split_name}.")

# === Run for all splits ===
if __name__ == "__main__":
    create_metadata("train-clean-100", "train")
    create_metadata("dev-clean", "dev")
    create_metadata("test-clean", "test")
