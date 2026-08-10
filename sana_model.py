import torch
from torch import nn
from sana_block import SANA_Block, modulate
from embeddings import PatchEmbeddings, TimestepEmbedder, CaptionEmbedder


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int = 2240, out_channels: int = 32, patch_size: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size)/hidden_size**0.5)  # scale and shift parameters for layer normalization     

    def forward(self, x, t):
        # x: [B, N, D]
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = self.final_layer(modulate(self.norm(x), shift=shift, scale=scale)) #(B, N, patch_size * patch_size * out_channels)
        return x

class SANAModel(nn.Module):
    def __init__(self, num_blocks: int = 12, hidden_size: int = 2240, head_dim: int = 32, mlp_ratio: float = 2.5, patch_size:int=1,
                 in_channels:int=32, max_model_length:int=300, caption_channels=2304):
        super().__init__()

        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.max_model_length = max_model_length

        self.patch_embeddings = PatchEmbeddings(in_channels=self.in_channels, patch_size=self.patch_size, embed_dim=self.hidden_size)
        self.timestep_embedder = TimestepEmbedder(hidden_size=self.hidden_size, freq_dim=256)
        self.caption_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=self.hidden_size, token_num=self.max_model_length)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.blocks = nn.ModuleList([
            SANA_Block(hidden_size=self.hidden_size, head_dim=self.head_dim, mlp_ratio=self.mlp_ratio) for _ in range(self.num_blocks)
        ])

        self.final_layer = FinalLayer(hidden_size=self.hidden_size, out_channels=self.in_channels, patch_size=self.patch_size)

    def unpatchify(self, x, h, w):
        # x: (B, N, patch_size * patch_size * out_channels) -> (B, C, H, W)
        # H, W: height and width of the original image
        B, N, D = x.shape
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, D).permute(0, 5, 1, 3, 2, 4)  # (B, D, H, patch_size, W, patch_size)
        x = x.reshape(B, D, h * self.patch_size, w * self.patch_size)  # (B, D, H*patch_size, W*patch_size)
        return x


    def forward(self, x, y, t, mask=None):
        # x: (B, C, H, W)
        # y: (B, N, L)
        # t: (B,D)
        # mask: (B, N) or None
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        x = self.patch_embeddings(x)  # (B, N, hidden_size)
        t_0 = self.timestep_embedder(t)  # (B, hidden_size)
        y = self.caption_embedder(y)  # (B, N, hidden_size)
        t = self.t_block(t_0)  # (B, 6 * hidden_size)
        for block in self.blocks:
            x = block(x, y, t, mask, (h, w))
        
        x = self.final_layer(x, t_0)  # (B, N, patch_size * patch_size * out_channels)

        x = self.unpatchify(x, h, w)  # (B, out_channels, H, W)
        return x

# ----------------------------------------------------------------------------- #
# factory + checkpoint loader
# ----------------------------------------------------------------------------- #
def SANAModel_1600M_P1_D20(**kw):
    return SANAModel(num_blocks=20, hidden_size=2240, patch_size=1, head_dim=32, **kw)


# def SANAModel_600M_P1_D28(**kw):
#     return SANAModel(num_blocks=28, hidden_size=1152, patch_size=1, head_dim=16, **kw)


def load_pth(model, path, load_ema=False):
    """
    Load an official NVlabs SANA `.pth` into `model`, mirroring the reference
    inference loader: unwrap `state_dict`, drop `pos_embed`, load non-strict.
    Returns (missing_keys, unexpected_keys).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if load_ema and isinstance(ckpt, dict) and "state_dict_ema" in ckpt:
        sd = ckpt["state_dict_ema"]
    elif isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
    else:
        sd = ckpt
    sd = {k: v for k, v in sd.items() if k not in ("pos_embed", "base_model.pos_embed", "model.pos_embed")}
    return model.load_state_dict(sd, strict=False)

if __name__ == "__main__":
    # sanity check: load the official 1.6B checkpoint and run a forward pass
    model = SANAModel_1600M_P1_D20()
    load_pth(model, "SANA1.5_1.6B_1024px.pth", load_ema=True)
    model.eval()
    B, C, H, W = 1, 32, 32, 32
    L, caption_channels = 300, 2304
    x = torch.randn(B, C, H, W)
    timestep = torch.randint(0, 1000, (B,))
    y = torch.randn(B, L, caption_channels)
    mask = torch.ones(B, L)
    with torch.no_grad():
        out = model(x, y, timestep, mask)
    print("Output shape:", out.shape)  # should be (B, C, H, W)
