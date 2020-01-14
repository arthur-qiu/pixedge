import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurNet(nn.Module):
    def __init__(self, factor = 2):
        super(BlurNet, self).__init__()

        filter_weight = torch.Tensor([[0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
                                      [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                                      [0.03832756, 0.05576627, 0.06319146, 0.05576627, 0.03832756],
                                      [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                                      [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684]])

        self.filter_weight1 = filter_weight.view(1, 1, 5, 5).repeat(3, 1, 1, 1).cuda()
        self.pad = nn.ReflectionPad2d((2, 2, 2, 2))
        self.factor = factor

    def forward(self, x):
        out = nn.functional.conv2d(self.pad(x), self.filter_weight1, groups = 3)
        out = nn.functional.interpolate(out, scale_factor=0.5, mode = 'bicubic')
        out = nn.functional.interpolate(out, scale_factor=2, mode = 'nearest')
        return out