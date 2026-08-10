import torch
from torch import nn
from feedforward import FeedForward
from attention import CrossAttention, LiteLA

def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x

class SANA_Block(nn.Module):
    def __init__(self, hidden_size: int = 2240, head_dim: int = 32, mlp_ratio: int = 2.5):
        super().__init__()
        self.cross_attention = CrossAttention(num_heads=hidden_size//head_dim, head_dim=head_dim, embed_dim=hidden_size)
        self.linear_attention = LiteLA(embed_dim=hidden_size, head_dim=head_dim)
        self.feed_forward = FeedForward(in_features=hidden_size, hidden_features=int(hidden_size*mlp_ratio))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size)/hidden_size ** 0.5)

    def forward(self,x,y,t,mask,HW):

        """
        x: [B, N, D]
        y: [B, 1, D] 
        t: [B, 6D]
        """

        B = x.size(0)
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = (self.scale_shift_table[None]* t.reshape(B,6,-1)).chunk(6, dim=1)
        x = x + gate_attn * self.linear_attention(modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + self.cross_attention(x,y,mask=mask)
        x = x + gate_ff * self.feed_forward(modulate(self.norm2(x), shift_ff, scale_ff), HW)
        return x
