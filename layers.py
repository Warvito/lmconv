import logging
import math
import pdb

import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import * 

logger = logging.getLogger("gen")

# DEBUG TO REMOVE WEIGHT NORMALIZATION
# from torch.nn.utils import weight_norm as wn
def wn(op):
    return op

def identity(x):
    return x

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out
    
    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1), 
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()
        
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down
        
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride, 
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1), 
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1), 
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()
        
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1), 
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, 
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, feature_norm_op=None, nonlinearity=concat_elu, skip_connection=0, dropout_prob=0.5):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu
        self.norm_input = feature_norm_op(num_filters) if feature_norm_op else identity
        
        if skip_connection != 0 : 
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else identity
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)
        self.norm_out = feature_norm_op(num_filters) if feature_norm_op else identity

    def forward(self, og_x, a=None, mask=None):
        x = self.conv_input(self.nonlinearity(og_x), mask=mask)
        x = self.norm_input(x)
        if a is not None : 
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x, mask=mask)
        a, b = torch.chunk(x, 2, dim=1)
        # TODO: Should x be normalized instead? or c3?
        a = self.norm_out(a)
        c3 = a * torch.sigmoid(b)
        return og_x + c3


class masked_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),
                 dilation=1, mask_type='B'):
        """2D Convolution with masked weight for autoregressive connection"""
        super(masked_conv2d, self).__init__()
        assert mask_type in ['A', 'B']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mask_type = mask_type

        # Pad to maintain spatial dimensions
        self.padding = ((dilation * (kernel_size[0] - 1)) // 2,
                   (dilation * (kernel_size[1] - 1)) // 2)

        # Conv parameters
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1
        assert self.weight.size(0) == out_channels
        assert self.weight.size(1) == in_channels
        assert self.weight.size(2) == kernel_size[0]
        assert self.weight.size(3) == kernel_size[1]
        mask = torch.ones(out_channels, in_channels, kernel_size[0], kernel_size[1])
        yc = kernel_size[0] // 2
        xc = kernel_size[1] // 2
        mask[:, :, yc, xc:] = 0
        mask[:, :, yc + 1:] = 0
        if mask_type == 'B':
            mask[:, :, yc, xc] = 1
        self.register_buffer('mask', mask)

        self.reset_parameters()

    def reset_parameters(self):
        # From PyTorch _ConvNd implementation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, padding=self.padding, dilation=self.dilation)


class input_masked_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, bias=True):
        super(input_masked_conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Pad to maintain spatial dimensions
        pad0 = (dilation * (kernel_size[0] - 1)) // 2
        pad1 = (dilation * (kernel_size[1] - 1)) // 2
        self.padding = (pad0, pad1)
        
        # Conv parameters
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # From PyTorch _ConvNd implementation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, mask=None):
        assert len(x.shape) == 4, "Unfold/fold only support 4D batched image-like tensors"
        output_shape = (x.shape[2], x.shape[3])
        C = x.shape[1]

        x = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding)

        if mask is not None:
            # Option 1: Repeat mask from 1x(k*k)xnum_patches to 1x(C*k*k)xnum_patches
            # mask_repeat = mask.repeat(1, x.size(1), 1)
            # x_unf = x_unf * mask_repeat

            # Option 3: Avoid repeating mask in_channels times by reshaping x_unf (memory efficient)
            assert x.size(1) % C == 0
            x_unf_channels_batched = x.view(x.size(0) * C, x.size(1) // C, x.size(2))
            x = torch.mul(x_unf_channels_batched, mask).view(x.shape)
        else:
            logger.warning("input_masked_conv2d not provided mask, not masking!")

        # Perform convolution via matrix multiplication and addition
        weight_matrix = self.weight.view(self.out_channels, -1)
        x = torch.einsum('ij,bjk->bik', weight_matrix, x)
        # x = weight_matrix.matmul(x)

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(2)

        # Output shape is preserved due to input padding
        return F.fold(x, output_shape, (1, 1))
