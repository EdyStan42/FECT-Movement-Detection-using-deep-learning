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
        # Padding must be (kernel_size - 1) * dilation // 2 to stay 'same'
        padding = (15 - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class RawSignalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(2, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(4)
        self.enc2 = ResidualBlock(16, 32, dilation=4)
        self.pool2 = nn.MaxPool1d(4)

        self.bottleneck = ResidualBlock(32, 64, dilation=8)

        # Upsampling
        self.up2 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec1 = ResidualBlock(32 + 16, 16)

        self.final = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        # Using concatenation for true U-Net behavior
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


# --- 2. THE IMPROVED LOADER ---
class RawPatientLoader:
    def __init__(self, sig_paths, mask_paths):
        self.sig_paths = sig_paths
        self.mask_paths = mask_paths

    def get_batches(self, window_size, stride, batch_size, shuffle=True):
        indices = np.arange(len(self.sig_paths))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()

            # Z-Score Normalization is usually better for ECG than Min-Max
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            # RMS Envelope
            squared = np.power(sig, 2)
            window = np.ones(500) / 500
            env = np.sqrt(np.convolve(squared, window, mode='same'))

            x_slices, y_slices = [], []
            for start in range(0, len(sig) - window_size, stride):
                sig_chunk = sig[start:start + window_size]
                env_chunk = env[start:start + window_size]
                mask_chunk = mask[start:start + window_size]

                x_slices.append(np.stack([sig_chunk, env_chunk], axis=0))
                y_slices.append(mask_chunk)

                if len(x_slices) == batch_size:
                    yield (torch.from_numpy(np.array(x_slices)).float(),
                           torch.from_numpy(np.array(y_slices)).float().unsqueeze(1))
                    x_slices, y_slices = [], []

            # Catch the remainder of the patient's data
            if len(x_slices) > 0:
                yield (torch.from_numpy(np.array(x_slices)).float(),
                       torch.from_numpy(np.array(y_slices)).float().unsqueeze(1))


# --- 3. MAIN TRAINING ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    WINDOW_SIZE = 5000  # 10s context (Better for movement)
    STRIDE = 500     # 90% overlap
    BATCH_SIZE = 8       # Increased for stability
    EPOCHS = 400
    LEARNING_RATE = 0.00001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))

    # --- THE VALIDATION SPLIT ---
    split = int(0.8 * len(sig_paths))
    train_loader = RawPatientLoader(sig_paths[:split], mask_paths[:split])
    val_loader = RawPatientLoader(sig_paths[split:], mask_paths[split:])

    model = RawSignalUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # The 20x weight that successfully caught the movement in your plot
    pos_weight = torch.tensor([20.0]).to(device)

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
            t_loss += loss.item(); t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight).item()
                v_steps += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss/t_steps:.6f} | Val Loss: {v_loss/v_steps:.6f}", flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"patient_centric_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()