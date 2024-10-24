import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MaskBlock(nn.Module):
    def __init__(self, embed_dim=64):
        super(MaskBlock, self).__init__()
        self.act = nn.ReLU(True)
        self.conv_head = default_conv(3, embed_dim, 3)

        self.conv_self = default_conv(embed_dim, embed_dim, 1)

        self.conv1 = default_conv(embed_dim, embed_dim, 3)
        self.conv1_1 = default_conv(embed_dim, embed_dim, 1)
        self.conv1_2 = default_conv(embed_dim, embed_dim, 1)
        self.conv_tail = default_conv(embed_dim, 3, 3)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv_self(x)
        x = x.mul(x)
        x = self.act(self.conv1(x))
        x = self.conv1_1(x).mul(self.conv1_2(x))

        return self.conv_tail(x)