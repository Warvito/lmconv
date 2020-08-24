import logging

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn

from utils import concat_elu

logger = logging.getLogger("gen")


def identity(x, *extra_args, **extra_kwargs):
    return x


class nin(nn.Module):
    def __init__(self, dim_in, dim_out, weight_norm=True):
        super(nin, self).__init__()
        if weight_norm:
            self.lin_a = wn(nn.Linear(dim_in, dim_out))
        else:
            self.lin_a = nn.Linear(dim_in, dim_out)
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, feature_norm_op=None, nonlinearity=concat_elu, skip_connection=0,
                 dropout_prob=0.5):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # cuz of concat elu
        self.norm_input = feature_norm_op(num_filters) if feature_norm_op else identity

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else identity
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)
        self.norm_out = feature_norm_op(num_filters) if feature_norm_op else identity

    def forward(self, og_x, a=None, mask=None):
        x = self.conv_input(self.nonlinearity(og_x), mask=mask)
        x = self.norm_input(x, mask=mask)
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x, mask=mask)
        a, b = torch.chunk(x, 2, dim=1)
        a = self.norm_out(a, mask=mask)
        c3 = a * torch.sigmoid(b)
        return og_x + c3


def pono(x, epsilon=1e-5):
    """Positional normalization"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std


class PONO(nn.Module):
    def forward(self, x, mask=None):
        # NOTE: mask argument is unused
        x, _, __ = pono(x)
        return x
