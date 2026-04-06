import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os


# --- 1. LOSS FUNCTION (BCE + DICE) ---
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        self.pos_weight = self.pos_weight.to(inputs.device)
        BCE = F.binary_cross_entropy(inputs, targets, weight=None, reduction='mean')
        return BCE + dice_loss


# --- 2. TRANSFORMER COMPONENTS ---
class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=4, num_layers=2, seq_len=320):
        super().__init__()
        # Standard Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional Encoding: Tells the brain "where" in the 10s window each beat is
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, channels))

    def forward(self, x):
        # x: [Batch, Channels, Seq] -> [B, 64, 320]
        x = x.permute(0, 2, 1)  # To [B, 320, 64]
        x = x + self.pos_emb
        x = self.transformer(x)
        return x.permute(0, 2, 1)  # Back to [B, 64, 320]


# --- 3. U-NET ARCHITECTURE ---
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


class RawSignalTransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (The Eyes)
        self.enc1 = ResidualBlock(1, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(4)
        self.enc2 = ResidualBlock(16, 32, dilation=4)
        self.pool2 = nn.MaxPool1d(4)

        # Bottleneck (The Brain)
        # 5120 / 4 / 4 = 320 sequence length
        self.bottleneck = TransformerBottleneck(32, num_heads=4, seq_len=320)
        self.bottleneck_proj = nn.Conv1d(32, 64, 1)  # Match decoder depth

        # Decoder
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Transformer bottleneck
        b = self.bottleneck(p2)
        b = self.bottleneck_proj(b)

        d2 = F.interpolate(b, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


# --- 4. DATA LOADER ---
class RawPatientLoader:
    def __init__(self, sig_paths, mask_paths):
        self.sig_paths = sig_paths
        self.mask_paths = mask_paths

    def get_batches(self, window_size, stride, batch_size, shuffle=True):
        indices = np.arange(len(self.sig_paths))
        if shuffle: np.random.shuffle(indices)

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            x_slices, y_slices = [], []
            for start in range(0, len(sig) - window_size, stride):
                sig_chunk = sig[start:start + window_size][np.newaxis, :]
                mask_chunk = mask[start:start + window_size]
                x_slices.append(sig_chunk)
                y_slices.append(mask_chunk)

                if len(x_slices) == batch_size:
                    yield (torch.from_numpy(np.array(x_slices)).float(),
                           torch.from_numpy(np.array(y_slices)).float().unsqueeze(1))
                    x_slices, y_slices = [], []


# --- 5. MAIN TRAINING LOOP ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    WINDOW_SIZE, STRIDE, BATCH_SIZE = 5120, 512, 16
    EPOCHS, LR = 400, 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))

    split = int(0.8 * len(sig_paths))
    train_loader = RawPatientLoader(sig_paths[:split], mask_paths[:split])
    val_loader = RawPatientLoader(sig_paths[split:], mask_paths[split:])

    model = RawSignalTransUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = DiceBCELoss(pos_weight=2.0)

    best_val_loss = float('inf')

    print("Starting TransUNet Training...")
    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad();
            out = model(x);
            loss = criterion(out, y)
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
            optimizer.step()
            t_loss += loss.item();
            t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, False):
                x, y = x.to(device), y.to(device)
                out = model(x);
                v_loss += criterion(out, y).item();
                v_steps += 1

        avg_val_loss = v_loss / v_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {t_loss / t_steps:.4f} | Val Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), "trans_unet_last.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "trans_unet_best.pth")
            print(f"--> Saved New Best TransUNet Model")


if __name__ == "__main__":
    main()