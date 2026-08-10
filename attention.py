from torch import nn
import torch
from torch.nn import functional as F

class CrossAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim:int, embed_dim:int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, condition, mask=None):
        B, N, C = x.shape
        Bc, M, Cc = condition.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3) #(B,H,N,D)
        kv = self.kv(condition).reshape(Bc,M,2,self.num_heads, self.head_dim).permute(2,0,3,1,4) #(2,B,H,M,D)
        k,v = kv[0], kv[1]

        attn_mask = None
        if mask is not None:
            attn_mask = (1 - mask.to(q.dtype))[:, None, None] * -10000.0  # (B,1,1,L)

        attn_out = F.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask) #(B,H,N,D)

        out = self.proj(attn_out.transpose(2,1).reshape(B,N,C))

        return out
    
class LiteLA(nn.Module):
    def __init__(self, embed_dim:int, head_dim: int, eps=1e-8):
        super ().__init__()
        self.head_dim = head_dim
        self.num_heads = embed_dim // head_dim
        self.eps = eps
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.kernel = nn.ReLU()

    def forward(self,x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)
        q = self.kernel(q)  # (B, num_heads, N, head_dim)
        k = self.kernel(k)  # (B, num_heads, N, head_dim)

        v = F.pad(v, (0,1), value=1.0, mode='constant')  # (B, num_heads, N, head_dim+1)

        vk = torch.matmul(k.transpose(-2,-1), v)  # (B, num_heads, head_dim, head_dim+1)

        out = torch.matmul(q, vk)  # (B, num_heads, N, head_dim+1)

        out = out[..., :-1] / (out[..., -1:] + self.eps)  # (B, num_heads, N, head_dim)
        
        out = out.transpose(1,2).reshape(B, N, C)  # (B, N, embed_dim)
        out = self.proj(out)  # (B, N, embed_dim)
        return out
