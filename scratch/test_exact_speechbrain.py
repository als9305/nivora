import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Official SpeechBrain building blocks
class SpeechBrainConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class SpeechBrainBatchNorm1d(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        return self.norm(x)


class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=nn.ReLU, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = SpeechBrainConv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, groups=groups)
        self.act = activation()
        self.norm = SpeechBrainBatchNorm1d(out_channels)

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = SpeechBrainConv1d(in_channels, se_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SpeechBrainConv1d(se_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.scale = scale
        self.blocks = nn.ModuleList([
            TDNNBlock(in_channel, hidden_channel, kernel_size, dilation)
            for _ in range(scale - 1)
        ])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        y = [chunks[0]]
        for i in range(1, self.scale):
            if i == 1:
                y_i = self.blocks[i - 1](chunks[i])
            else:
                y_i = self.blocks[i - 1](chunks[i] + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SERes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TDNNBlock(in_channels, out_channels, 1, 1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, 1, 1)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

    def forward(self, x):
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = SpeechBrainConv1d(attention_channels, channels, 1)

    def forward(self, x):
        L = x.shape[-1]
        mean_global = x.mean(dim=2, keepdim=True)
        var_global = (x - mean_global).pow(2).mean(dim=2, keepdim=True)
        std_global = torch.sqrt(var_global.clamp(min=1e-12))

        attn_in = torch.cat([x, mean_global.repeat(1, 1, L), std_global.repeat(1, 1, L)], dim=1)
        attn = self.conv(self.tanh(self.tdnn(attn_in)))
        attn = F.softmax(attn, dim=2)

        mean_weighted = (attn * x).sum(dim=2)
        var_weighted = (attn * (x - mean_weighted.unsqueeze(2)).pow(2)).sum(dim=2).clamp(min=1e-12)
        std_weighted = torch.sqrt(var_weighted)

        pooled_stats = torch.cat((mean_weighted, std_weighted), dim=1).unsqueeze(2)
        return pooled_stats


class ExactECAPATDNN(nn.Module):
    def __init__(self, input_size=80, channels=[1024, 1024, 1024, 1024, 3072], lin_neurons=192):
        super().__init__()
        self.blocks = nn.ModuleList([
            TDNNBlock(input_size, channels[0], 5, 1),
            SERes2NetBlock(channels[0], channels[1], res2net_scale=8, se_channels=128, kernel_size=3, dilation=2),
            SERes2NetBlock(channels[1], channels[2], res2net_scale=8, se_channels=128, kernel_size=3, dilation=3),
            SERes2NetBlock(channels[2], channels[3], res2net_scale=8, se_channels=128, kernel_size=3, dilation=4),
        ])
        self.mfa = TDNNBlock(channels[0] * 3, channels[-1], 1, 1)
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels=128)
        self.asp_bn = SpeechBrainBatchNorm1d(channels[-1] * 2)
        self.fc = SpeechBrainConv1d(channels[-1] * 2, lin_neurons, 1)

    def forward(self, x):
        # x: [B, 80, T]
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)

        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.asp_bn(x)
        x = self.fc(x)
        x = x.squeeze(2)
        return F.normalize(x, p=2, dim=-1)


def test_exact_model():
    model = ExactECAPATDNN()
    ckpt = torch.load("models/ecapa_voxceleb.ckpt", map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("ExactECAPATDNN successfully loaded state_dict with strict=True!")

    # Test with different inputs
    import huggingface_hub, pandas as pd
    p15 = huggingface_hub.hf_hub_download(
        repo_id='DynamicSuperb/SpoofDetection_ASVspoof2015',
        filename='data/test-00000-of-00001-4a252b1e92dc86af.parquet',
        repo_type='dataset'
    )
    df15 = pd.read_parquet(p15)
    auth_samples = df15[df15['label'] == 'authentic'].head(5)

    sys.path.insert(0, os.path.abspath("."))
    from models.voice_auth import AudioPreprocessor, AudioFeatureExtractor

    embs = []
    for i in range(len(auth_samples)):
        b = auth_samples.iloc[i]['audio']['bytes']
        sig, sr, _ = AudioPreprocessor.decode_wav_bytes(b)
        feats = AudioFeatureExtractor.extract_features(sig, sr)
        lmel = feats['log_mel_80']
        # CMS
        lmel = lmel - np.mean(lmel, axis=1, keepdims=True)
        t = torch.from_numpy(lmel).float().unsqueeze(0)
        with torch.no_grad():
            emb = model(t).squeeze(0).numpy()
        embs.append(emb)

    print("\nPairwise Cosine Similarities across 5 DIFFERENT authentic speakers:")
    for i in range(len(embs)):
        row = [round(float(np.dot(embs[i], embs[j])), 4) for j in range(len(embs))]
        print(f"Sample {i}: {row}")

if __name__ == "__main__":
    test_exact_model()
