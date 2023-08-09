import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, AdaptiveAvgPool2d, BatchNorm2d

def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    assert num_channels % groups == 0, 'num_channels should be divisible by groups'
    channels_per_group = num_channels // groups
    x = torch.reshape(
        input=x, shape=(batch_size, groups, channels_per_group, height, width))
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = torch.reshape(x, (batch_size, num_channels, height, width))
    return x


"""
Squeeze-and-Excitation Networks module
https://arxiv.org/abs/1709.01507
"""
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(output_size=1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            )
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            )
        torch.nn.init.torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv1.bias, 0.)
        torch.nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, inputs):
        # squeeze
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        # excitation
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return torch.multiply(inputs, outputs)


"""
ShuffleNet v2 的组成部分，conv+batchnorm+hardswish
"""
class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        nn.init.kaiming_normal_(self._conv.weight)

        self._batch_norm = BatchNorm2d(
            out_channels)
            # 既然coefficient都是0,此处L2Decay不管了, yty
            #weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            #bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if act == "hard_swish":
            act = 'hardswish'
        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act:
            y = getattr(F, self.act)(y)
        return y


"""
Esnet basic block of 2 versions, see paper at:https://arxiv.org/abs/2111.00902
"""
class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


class ESNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 act="hard_swish",
                 feature_maps=[4, 11, 14],
                 channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super(ESNet,self).__init__()
        self.scale = scale
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1, 24, make_divisible(128 * scale), make_divisible(256 * scale),
            make_divisible(512 * scale), 1024
        ]
        self._out_channels=[]
        self._feature_idx=0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        self._block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    self.add_module(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        module=InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act))
                    block = self.get_submodule(str(stage_id + 2) + '_' + str(i + 1))
                else:
                    self.add_module(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        module=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act))
                    block = self.get_submodule(str(stage_id + 2) + '_' + str(i + 1))
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self,inputs):
        images, _ = inputs
        images = torch.stack(images,dim=0)
        # stack image as a tensor
        #y = self._conv1(inputs['image'])
        y = self._conv1(images)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)

        return outs




