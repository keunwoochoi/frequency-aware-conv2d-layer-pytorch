import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_freq_map(min_freq, max_freq, num_freq, dtype=torch.float32):
    """Given params, it returns a frequency map.
    num_freq should be positive integer.
    """
    if num_freq > 1:
        step = float(max_freq - min_freq) / (num_freq - 1)
        map = torch.arange(start=min_freq,
                           end=max_freq + step,
                           step=step,
                           dtype=dtype)
        return torch.reshape(map, (1, 1, -1, 1))
    elif num_freq == 1:
        return torch.tensor([float(max_freq + min_freq) / 2]).view([1, 1, -1, 1])
    else:
        raise ValueError('num_freq should be positive but we got: {}'.format(num_freq))


class FreqAwareConv2dLinearBiasOffset(nn.Conv2d):
    """A modified conv2d layer that concats the frequency map along the channel axis.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
                 ):
        super(FreqAwareConv2dLinearBiasOffset, self).__init__(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        self.freq_map = None
        self.min_freq = 0.0
        self.max_freq = 1.0
        self.freq_axis = 2
        self.ch_axis = 1  # it follows torch convention, hence not allowing to change it.

    def forward(self, input: torch.tensor):
        """

        Maybe the input shape is (batch, ch, freq, time),
        ..or whatever as long as the freq axis index == self.freq_axis

        Also assumes the input size is always the same so that it can reuse the
        same freq_map

        """
        if self.freq_map is None:
            num_freq = input.shape[self.freq_axis]
            self.freq_map = _get_freq_map(self.min_freq, self.max_freq, num_freq,
                                          dtype=input.dtype).to(input.device)

        expand_shape = list(input.shape)
        expand_shape[self.ch_axis] = 1
        expanded_map = self.freq_map.expand(*expand_shape)

        input = torch.cat((input, expanded_map),
                          dim=self.ch_axis)
        return super(FreqAwareConv2dLinearBiasOffset, self).forward(input)

