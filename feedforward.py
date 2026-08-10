import torch 
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, activation=nn.SiLU()):
        super().__init__()
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.activation = activation

        self.inverted_conv = nn.Conv2d(in_features, 2*hidden_features, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(2*hidden_features, 2*hidden_features, kernel_size=3, padding=1, groups=2*hidden_features)
        self.pointwise_conv = nn.Conv2d(hidden_features, in_features, kernel_size=1)

    def forward(self, x, HW):
        """
        x: [B, N, D]
        """
        H,W = HW
        B, N, D = x.shape
        x = x.reshape(B,H,W,D).permute(0,3,1,2)  # (B, D, H, W)
        x = self.inverted_conv(x)
        x = self.activation(x)
        x = self.depthwise_conv(x)
        a,g = x.chunk(2, dim=1)
        x = a * self.activation(g)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = x.reshape(B, D, N).permute(0, 2, 1)         # back to (B, N, C)
        return x
