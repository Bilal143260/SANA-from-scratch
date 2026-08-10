import torch
import math
from torch import nn

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        half = embedding_dim // 2
        freq_range = torch.arange(0, half, dtype=torch.float32) / half
        max_period = 10000.0
        freqs = torch.exp(-math.log(max_period) * freq_range)
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    
    def forward(self, timesteps):
        emb = self.mlp(self.get_timestep_embedding(timesteps, self.freq_dim))
        return emb  
    
class CaptionEmbedder(nn.Module):
    """Projects Gemma embeddings (caption_channels -> hidden). Holds the null-caption buffer."""

    def __init__(self, in_channels, hidden_size, token_num):
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size)
        )
        # registered as a buffer named `y_embedding` (loaded from the checkpoint)
        self.register_buffer("y_embedding",
                             torch.randn(token_num, in_channels) / in_channels ** 0.5) #it stores the null caption embedding, register_buffer stores non trainable entity in the model, so that it can be saved and loaded with the model's state_dict

    def forward(self, caption):
        return self.y_proj(caption)
