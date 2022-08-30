import torch
from torch import nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    
    def __init__(self, in_channels, drop_rate=0.1):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1, padding=0),
            # nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask.unsqueeze(dim=1), size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        avg_pool = F.adaptive_avg_pool2d(self.mask(Fs, Ys), (1, 1))
        # max_pool = F.adaptive_max_pool2d(self.mask(Fs, Ys), (1, 1))
        # attn = torch.cat([avg_pool, max_pool],dim=1)
        g = self.gate(avg_pool)
        Fs = g * Fs
        return Fs


# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.flatten(start_dim=1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )
#         self.pool_types = pool_types

#     @staticmethod
#     def mask(embedding, mask):
#         h, w = embedding.size()[-2:]
#         mask = F.interpolate(mask.unsqueeze(dim=1), size=(h, w), mode='nearest')
#         mask=mask
#         return mask * embedding


#     def forward(self, *x):
#         Fs, Ys = x
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type == 'avg':
#                 avg_pool = F.adaptive_avg_pool2d(self.mask(Fs, Ys), (1, 1))
#                 channel_att_raw = self.mlp(avg_pool)
#             elif pool_type == 'max':
#                 max_pool = F.adaptive_max_pool2d(self.mask(Fs, Ys), (1, 1))
#                 channel_att_raw = self.mlp(max_pool)

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         channel_att_sum = torch.sigmoid(channel_att_sum)
#         scale = channel_att_sum.unsqueeze(2).unsqueeze(3)
#         return Fs * scale


    
# class Attention(nn.Module):
#     """
#     Guided Attention Module (GAM).
#     Args:
#         in_channels: interval channel depth for both input and output
#             feature map.
#         drop_rate: dropout rate.
#     """

#     def __init__(self, in_channels, drop_rate=0.5):
#         super().__init__()
#         self.DEPTH = in_channels
#         self.DROP_RATE = drop_rate
#         self.gate = nn.Sequential(
#             nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
#                       kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
#                       kernel_size=1),
#             nn.Dropout(p=drop_rate),
#             nn.Sigmoid())

#     @staticmethod
#     def mask(embedding, mask):
#         h, w = embedding.size()[-2:]
#         mask = F.interpolate(mask.unsqueeze(dim=1), size=(h, w), mode='nearest')
#         mask=mask
#         return mask * embedding

#     def forward(self, *x):
#         Fs, Ys = x
#         att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
#         g = self.gate(att)
#         Fs = g * Fs
#         return Fs
