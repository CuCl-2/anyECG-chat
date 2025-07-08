import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., conv_patch = False):
        super().__init__()
        # ecg(12, 1000)
        # patch (1, 200) 12*5=60patch
        self.dim = dim
        patch_dim = patch_size[0] * patch_size[1]
        num_patches = int(seq_len * channels / patch_size[0] / patch_size[1])

        if conv_patch:
            assert patch_size[0] == 12, 'only temporal patching is valid when using conv_patch'
            self.to_patch_embedding = nn.Sequential(
                nn.Conv1d(channels, dim//4, kernel_size=15, stride=1, padding=7),
                nn.BatchNorm1d(dim//4),
                nn.ReLU(),
                nn.Conv1d(dim//4, dim//2, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(dim//2),
                nn.ReLU(),
                nn.Conv1d(dim//2, dim, kernel_size=patch_size[1], stride=patch_size[1]),
                Rearrange('b c n -> b n c'),
            )
        else:            
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b (c p1) (n p2) -> b (c n) (p1 p2)', p1 = patch_size[0], p2 = patch_size[1]),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        self.lead_embedding = nn.Parameter(torch.randn(1, channels, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, int(seq_len/patch_size[1]) + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, series):
        x = self.to_patch_embedding(series)
        x = rearrange(x, 'b (c n) d -> b c n d', c = 12)
        x += self.lead_embedding.unsqueeze(2) + self.pos_embedding.unsqueeze(1)[:, :, 1:, :]
        x = rearrange(x, 'b c n d -> b (c n) d')
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x[:, :1, :] += self.pos_embedding[:, :1, :]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.fc(self.norm(cls_tokens))
    
    def encode(self, series):
        x = self.to_patch_embedding(series)
        x = rearrange(x, 'b (c n) d -> b c n d', c = 12)
        x += self.lead_embedding.unsqueeze(2) + self.pos_embedding.unsqueeze(1)[:, :, 1:, :]
        x = rearrange(x, 'b c n d -> b (c n) d')
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x[:, :1, :] += self.pos_embedding[:, :1, :]
        x = self.dropout(x)

        x = self.transformer(x)
        return x

def ecg_vit(num_classes = 3, patch_size = (1, 200), heads = 8, conv_patch = False, seq_len = 1000):
    dim = patch_size[0] * patch_size[1]
    print('patch_dim', dim)
    dim_valid = [64, 128, 256, 512, 1024, 2048, 4096]
    diff = [abs(dim - item) for item in dim_valid]
    min_diff_index = diff.index(min(diff))
    dim = dim_valid[min_diff_index]
    print('model_dim', dim)
    dim_head = int(dim / heads)
    mlp_dim = dim * 2
    return ViT(
        seq_len = seq_len,
        patch_size = patch_size,
        channels=12,
        num_classes = num_classes,
        depth = 6,
        heads = heads,
        dim = dim,
        dim_head= dim_head,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        conv_patch=conv_patch
    )

def ecg_vit_base(num_classes = 512, patch_size = (1, 200), heads = 12, conv_patch = False, seq_len = 1000):
    dim = patch_size[0] * patch_size[1]
    # print('patch_dim', dim)
    dim=768
    # print('model_dim', dim)
    dim_head = int(dim / heads)
    mlp_dim = dim * 4
    return ViT(
        seq_len = seq_len,
        patch_size = patch_size,
        channels=12,
        num_classes = num_classes,
        depth = 12,
        heads = heads,
        dim = dim,
        dim_head= dim_head,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        conv_patch=conv_patch
    )

def ecg_vit_large(num_classes = 512, patch_size = (1, 200), heads = 16, conv_patch = False, seq_len = 1000):
    dim = patch_size[0] * patch_size[1]
    print('patch_dim', dim)
    dim=1024
    print('model_dim', dim)
    dim_head = int(dim / heads)
    mlp_dim = dim * 4
    return ViT(
        seq_len = seq_len,
        patch_size = patch_size,
        channels=12,
        num_classes = num_classes,
        depth = 24,
        heads = heads,
        dim = dim,
        dim_head= dim_head,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        conv_patch=conv_patch
    )

if __name__ == '__main__':
    # 写一个100hz (12, 1000)分成（1， 250）patch共12*4=48个patch
    model = ecg_vit(num_classes=5, patch_size=(1, 200), conv_patch= False).cuda()
    time_series = torch.randn(3, 12, 1000).cuda()
    logits = model(time_series) # (4, 1000)
    print(logits.shape)

