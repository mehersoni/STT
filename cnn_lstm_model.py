import torch.nn as nn

CHAR_TO_INDEX = {
    "<BLANK>": 0,
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
    "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14,
    "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21,
    "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26, " ": 27, "'": 28
}
INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}

class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),      # 80 → 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=256,
            num_layers=2, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(256 * 2, vocab_size)

    def forward(self, x):
        x = x.transpose(1, 2)       # [B, 80, T] → [B, T, 128]
        x = self.cnn(x)             # [B, 128, T]
        x = x.transpose(1, 2)       # [B, T, 128]
        x, _ = self.lstm(x)         # [B, T, 512]
        x = self.classifier(x)      # [B, T, V]
        return x
