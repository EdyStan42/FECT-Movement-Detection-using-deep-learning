import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os


# --- 1. ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = (15 - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding, dilation=dilation),
            # CHANGED: Swapped BatchNorm for GroupNorm for validation stability
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class RawSignalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16, dilation=1)
        self.pool = nn.MaxPool1d(4)
        self.enc2 = ResidualBlock(16, 32, dilation=4)
        self.bottleneck = ResidualBlock(32, 64, dilation=8)

        # We use F.interpolate in forward, so these stay as feature extractors
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        b = self.bottleneck(p2)

        # Use concatenation and forced interpolation to avoid size mismatches
        d2 = F.interpolate(b, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


# --- 2. THE LOADER ---
class RawPatientLoader:
    def __init__(self, sig_paths, mask_paths):
        self.sig_paths = sig_paths
        self.mask_paths = mask_paths

    def get_batches(self, window_size, stride, batch_size, is_training=True):
        indices = np.arange(len(self.sig_paths))
        if is_training: np.random.shuffle(indices)

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            x_patient, y_patient = [], []
            for start in range(0, len(sig) - window_size, stride):
                # Ensure input is [1, Window]
                x_patient.append(sig[start:start + window_size][np.newaxis, :])
                y_patient.append(mask[start:start + window_size])

                if len(x_patient) == batch_size:
                    yield (torch.from_numpy(np.array(x_patient)).float(),
                           torch.from_numpy(np.array(y_patient)).float().unsqueeze(1))
                    x_patient, y_patient = [], []

            # Yield remainder
            if len(x_patient) > 0:
                yield (torch.from_numpy(np.array(x_patient)).float(),
                       torch.from_numpy(np.array(y_patient)).float().unsqueeze(1))


# --- 3. MAIN ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    WINDOW_SIZE = 5120  # Switched to multiple of 16
    STRIDE = 512
    BATCH_SIZE = 16
    EPOCHS = 400
    LEARNING_RATE = 0.0001  # Slightly higher is okay with GroupNorm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))

    split = int(0.8 * len(sig_paths))
    train_loader = RawPatientLoader(sig_paths[:split], mask_paths[:split])
    val_loader = RawPatientLoader(sig_paths[split:], mask_paths[split:])

    model = RawSignalUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # REDUCED: 2.0 or 5.0 is much safer than 20.0 for validation stability
    pos_weight = torch.tensor([2.0]).to(device)

    best_v_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item();
            t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight).item()
                v_steps += 1

        avg_v = v_loss / v_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {t_loss / t_steps:.4f} | Val Loss: {avg_v:.4f}")

        # Last/Best Save logic
        torch.save(model.state_dict(), "model_last.pth")
        if avg_v < best_v_loss:
            best_v_loss = avg_v
            torch.save(model.state_dict(), "Original_RES_model_best.pth")


if __name__ == "__main__":
    main()
