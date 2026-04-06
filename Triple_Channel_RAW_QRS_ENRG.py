import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os
from scipy.signal import find_peaks


# --- 1. FEATURE EXTRACTION ---
def extract_clinical_features(sig):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    # CH1: R-Peak Amplitudes (Pulse Train)
    peaks, _ = find_peaks(sig_norm, height=0.5, distance=150)
    amp_map = np.zeros_like(sig_norm)
    amp_map[peaks] = sig_norm[peaks]

    # CH2: Rolling Energy (1-second window)
    # Using a slightly faster vectorized rolling window for energy
    energy_map = np.zeros_like(sig_norm)
    res = np.array([np.std(sig_norm[max(0, i - 250):min(len(sig_norm), i + 250)])
                    for i in range(0, len(sig_norm), 50)])
    energy_map = np.interp(np.arange(len(sig_norm)), np.arange(0, len(sig_norm), 50), res)

    return np.stack([sig_norm, amp_map, energy_map], axis=0)


# --- 2. ARCHITECTURE (GroupNorm + Transformer) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = (15 - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding, dilation=dilation),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, seq_len=320):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dim_feedforward=channels * 4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, channels))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.pos_emb
        x = self.transformer(x)
        return x.permute(0, 2, 1)


class SpecialistTransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(3, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(4)
        self.enc2 = ResidualBlock(16, 32, dilation=4)
        self.pool2 = nn.MaxPool1d(4)
        self.bottleneck = TransformerBottleneck(32, seq_len=320)
        self.bottleneck_proj = nn.Conv1d(32, 64, 1)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, x):
        e1 = self.enc1(x);
        p1 = self.pool1(e1)
        e2 = self.enc2(p1);
        p2 = self.pool2(e2)
        b = self.bottleneck(p2);
        b = self.bottleneck_proj(b)
        d2 = F.interpolate(b, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


# --- 3. DATA LOADER ---
class ClinicalPatientLoader:
    def __init__(self, sig_paths, mask_paths):
        self.sig_paths = sig_paths
        self.mask_paths = mask_paths

    def get_batches(self, window_size, stride, batch_size, shuffle=True):
        indices = np.arange(len(self.sig_paths))
        if shuffle: np.random.shuffle(indices)

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            full_feat = extract_clinical_features(sig)
            x_slices, y_slices = [], []
            for start in range(0, sig.shape[0] - window_size, stride):
                x_slices.append(full_feat[:, start:start + window_size])
                y_slices.append(mask[start:start + window_size])
                if len(x_slices) == batch_size:
                    yield (torch.from_numpy(np.array(x_slices)).float(),
                           torch.from_numpy(np.array(y_slices)).float().unsqueeze(1))
                    x_slices, y_slices = [], []


# --- 4. OPTIMIZED LOSS ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        # pos_weight=5.0 helps the model care more about the rare movement blocks
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, inputs, targets):
        # We use logits directly with BCEWithLogitsLoss for better numerical stability
        device = inputs.device
        return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight.to(device))


# --- 5. MAIN TRAINING LOOP ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    WINDOW_SIZE, STRIDE, BATCH_SIZE = 5120, 512, 16
    EPOCHS, LR = 400, 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))

    split = int(0.8 * len(sig_paths))
    train_loader = ClinicalPatientLoader(sig_paths[:split], mask_paths[:split])
    val_loader = ClinicalPatientLoader(sig_paths[split:], mask_paths[split:])

    model = SpecialistTransUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = WeightedBCELoss(pos_weight=5.0)

    best_val_loss = float('inf')

    print("Starting 3-Channel TransUNet Training...")
    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, shuffle=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item();
            t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, shuffle=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item();
                v_steps += 1

        avg_val_loss = v_loss / v_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {t_loss / t_steps:.4f} | Val Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), "3channel_trans_unet_last.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "3channel_trans_unet_best.pth")
            print(f"--> Saved New Best TransUNet Model")


if __name__ == "__main__":
    main()