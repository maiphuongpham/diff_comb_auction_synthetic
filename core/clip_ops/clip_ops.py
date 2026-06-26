from __future__ import absolute_import, division, print_function

import torch

# Please see clip_op functions for other setting in the repository of the `Optimal auctions through attention` paper


def clip_op_01(x):
    x.data.clamp_(0, 1)


def clip_op_12(x):
    x.data.clamp_(1, 2)


def clip_op_23(x):
    x.data.clamp_(2, 3)

def clip_op_416_47(x):
    min_val = torch.FloatTensor([4])[None, None, None, :].to(x.device)
    max_val = torch.FloatTensor([16, 7])[None, None, None, :].to(x.device)
    x.data.clamp_(min_val, max_val)

def clip_op_12_15(x):
    min_val = torch.FloatTensor([1])[None, None, :, None].to(x.device)
    max_val = torch.FloatTensor([2, 5])[None, None, :, None].to(x.device)
    x.data.clamp_(min_val, max_val)

def clip_op_gamma(x):
    x.data.clamp_(0, 10)

def clip_op_cc2(x):
    x.data.clamp_(0, 1)

def clip_op_airport(x):
    """
    Clip operation for airport data.
    The values are expected to be in the range [0, 1].
    """
    x.data.clamp_(0, 1)  # Assuming the maximum value in airport data is 50000