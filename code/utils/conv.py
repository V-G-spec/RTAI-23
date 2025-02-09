from typing import Tuple

import numpy as np
import torch


def flat(iterable) -> int:
    return np.product(iterable)


def get_hw_after_conv(conv, in_shape) -> Tuple[int, int]:
    return tuple([((in_shape[i] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i]) + 1 for i in [-2, -1]])


def torch_conv_layer_to_affine(conv: torch.nn.Conv2d, in_shape) -> torch.nn.Linear:
    """
    :param conv: Convolutional layer
    :param in_shape: Shape of input tensor
    :return: Fully connected layer

    This is how it was tested:
        filter_size, stride, padding = 3, 2, 1
        img = torch.rand((1, 2, 6, 7))
        conv = nn.Conv2d(2, 5, filter_size, stride=stride, padding=padding)
        fc = torch_conv_layer_to_affine(conv, img.shape)
        res1 = fc(img.flatten())
        res2 = conv(img).flatten()
        worst_error = (res1 - res2).abs().max()

    The worst_error was always close to 0, so it seems to work.
    """
    assert len(in_shape) == 3
    c_in, h_in, w_in = in_shape
    h_out, w_out = get_hw_after_conv(conv, in_shape)
    out_shape = (conv.out_channels, h_out, w_out)
    fc = torch.nn.Linear(in_features=flat(in_shape), out_features=flat(out_shape), dtype=conv.weight.dtype)
    # Need this otherwise the weights are initialized randomly and we are not updating at all the indices
    fc.weight.data.fill_(0.0)

    with torch.no_grad():
        for co in range(conv.out_channels):
            xo, yo = torch.meshgrid(torch.arange(h_out), torch.arange(w_out), indexing='ij')

            # Calculate xi0, yi0 for all positions
            xi0 = (-conv.padding[0] + conv.stride[0] * xo).reshape(-1)
            yi0 = (-conv.padding[1] + conv.stride[1] * yo).reshape(-1)

            for xd in range(conv.kernel_size[0]):  # Iterate over kernel height/p
                for yd in range(conv.kernel_size[1]):  # Iterate over kernel width/q
                    valid_mask = (0 <= xi0 + xd) & (xi0 + xd < h_in) & (0 <= yi0 + yd) & (yi0 + yd < w_in)
                    valid_xi0 = xi0[valid_mask] + xd
                    valid_yi0 = yi0[valid_mask] + yd
                    valid_xo = xo.reshape(-1)[valid_mask]
                    valid_yo = yo.reshape(-1)[valid_mask]

                    cw = conv.weight[co, :, xd, yd].unsqueeze(1)  # Add dimension for broadcasting
                    index_weight = np.ravel_multi_index((co, valid_xo, valid_yo), out_shape)
                    for ci in range(conv.in_channels):
                        index_input = np.ravel_multi_index((ci, valid_xi0, valid_yi0), in_shape)
                        fc.weight[index_weight, index_input] = cw[ci]

            # Update bias for all xo, yo positions
            index_bias = np.ravel_multi_index((co, xo.reshape(-1), yo.reshape(-1)), out_shape)
            fc.bias[index_bias] = conv.bias[co]

    return fc
