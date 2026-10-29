import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.ModuleDict({'conv': nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)})
        self.norm = nn.ModuleDict({'norm': nn.BatchNorm1d(out_channels)})
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm['norm'](self.conv['conv'](x)))

class SEBlock(nn.Module):
    def __init__(self, in_channels, bottleneck=128):
        super().__init__()
        self.conv1 = nn.ModuleDict({'conv': nn.Conv1d(in_channels, bottleneck, 1)})
        self.relu = nn.ReLU()
        self.conv2 = nn.ModuleDict({'conv': nn.Conv1d(bottleneck, in_channels, 1)})
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = self.relu(self.conv1['conv'](s))
        s = self.sigmoid(self.conv2['conv'](s))
        return x * s

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        self.scale = scale
        self.width = out_channels // scale
        self.tdnn1 = Conv1dReluBn(in_channels, out_channels, 1)
        self.res2net_block = nn.ModuleDict({
            'blocks': nn.ModuleList([
                Conv1dReluBn(self.width, self.width, kernel_size, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
                for _ in range(scale - 1)
            ])
        })
        self.tdnn2 = Conv1dReluBn(out_channels, out_channels, 1)
        self.se_block = SEBlock(out_channels, bottleneck=128)

    def forward(self, x):
        residual = x
        x = self.tdnn1(x)
        chunks = torch.chunk(x, self.scale, dim=1)
        out = []
        for i in range(self.scale):
            if i == 0:
                out.append(chunks[i])
            elif i == 1:
                out.append(self.res2net_block['blocks'][i - 1](chunks[i]))
            else:
                out.append(self.res2net_block['blocks'][i - 1](chunks[i] + out[i - 1]))
        x = torch.cat(out, dim=1)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual

class ECAPATDNN(nn.Module):
    def __init__(self, in_channels=80, channels=1024, emb_dim=192):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dReluBn(in_channels, channels, 5, padding=2),
            Res2NetBlock(channels, channels, scale=8, kernel_size=3, dilation=2),
            Res2NetBlock(channels, channels, scale=8, kernel_size=3, dilation=3),
            Res2NetBlock(channels, channels, scale=8, kernel_size=3, dilation=4)
        ])
        self.mfa = Conv1dReluBn(channels * 3, channels * 3, 1)
        
        # Attentive Statistics Pooling
        self.asp = nn.ModuleDict({
            'tdnn': Conv1dReluBn(channels * 3 * 3, 128, 1),
            'conv': nn.ModuleDict({'conv': nn.Conv1d(128, channels * 3, 1)})
        })
        self.asp_bn = nn.ModuleDict({'norm': nn.BatchNorm1d(channels * 6)})
        self.fc = nn.ModuleDict({'conv': nn.Conv1d(channels * 6, emb_dim, 1)})

    def forward(self, x):
        # x: [B, 80, T]
        out0 = self.blocks[0](x)
        out1 = self.blocks[1](out0)
        out2 = self.blocks[2](out1)
        out3 = self.blocks[3](out2)

        mfa_in = torch.cat([out1, out2, out3], dim=1)
        mfa_out = self.mfa(mfa_in)

        # Temporal context expansion for ASP: [-1, 0, 1]
        t_m1 = torch.cat([mfa_out[:, :, :1], mfa_out[:, :, :-1]], dim=-1)
        t_p1 = torch.cat([mfa_out[:, :, 1:], mfa_out[:, :, -1:]], dim=-1)
        asp_in = torch.cat([t_m1, mfa_out, t_p1], dim=1)  # [B, 9216, T]

        attn = self.asp['tdnn'](asp_in)  # [B, 128, T]
        attn = torch.softmax(self.asp['conv']['conv'](attn), dim=-1)  # [B, 3072, T]
        
        mean = torch.sum(mfa_out * attn, dim=-1)  # [B, 3072]
        var = torch.sum((mfa_out ** 2) * attn, dim=-1) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        
        stats = torch.cat([mean, std], dim=-1).unsqueeze(-1)  # [B, 6144, 1]
        stats = self.asp_bn['norm'](stats)
        
        emb = self.fc['conv'](stats).squeeze(-1)  # [B, 192]
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

if __name__ == "__main__":
    ckpt_path = hf_hub_download('speechbrain/spkrec-ecapa-voxceleb', 'embedding_model.ckpt')
    state = torch.load(ckpt_path, map_location='cpu')
    model = ECAPATDNN(in_channels=80, channels=1024, emb_dim=192)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("STRICT LOAD AND MODEL EVAL SUCCESSFUL!")
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 80, 200)
        emb = model(dummy_input)
        print("Output embedding shape:", emb.shape)
        print("L2 Norm of embedding:", torch.norm(emb, p=2).item())
