import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import constant_init, kaiming_init, normal_init
from ..runner import load_checkpoint


STAGE_REPEATS = [4, 8, 4, 1]

# index 0 is invalid and should never be called.
# only used for indexing convenience.
STAGE_OUT_CHANNELS = {
    0.25: [-1, 12, 24, 48, 96, 512],
    0.5: [-1, 24, 48, 96, 192, 1024],
    1.0: [-1, 24, 116, 232, 464, 1024],
    1.5: [-1, 24, 176, 352, 704, 1024],
    2.0: [-1, 24, 224, 488, 976, 2048]
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetV2Block(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel, dilation=1):
        super(ShuffleNetV2Block, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1*dilation,
                          groups=oup_inc, dilation=dilation, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1*dilation, groups=inp,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1*dilation,
                          groups=oup_inc, dilation=dilation, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 width_mult=1.,
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 image_channel=3,
                 ):
        """

        :param num_stages:
        :param dilations:
        :param out_indices:
        :param width_mult:
        :param frozen_stages:
        """
        super(ShuffleNetV2, self).__init__()
        assert 1 <= num_stages <= len(STAGE_REPEATS)
        self.stage_repeats = STAGE_REPEATS

        self.out_indices = list(out_indices)
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        if width_mult in STAGE_OUT_CHANNELS.keys():
            self.stage_out_channels = STAGE_OUT_CHANNELS[width_mult]
        else:
            raise ValueError(""" Unsupported width_mult """)

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(image_channel, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = []
        # building inverted residual blocks
        for idxstage in range(num_stages):
            if idxstage < 3:
                numrepeat = self.stage_repeats[idxstage]
                output_channel = self.stage_out_channels[idxstage + 2]
                features = []
                for i in range(numrepeat):
                    if i == 0:
                        # inp, oup, stride, benchmodel):
                        features.append(ShuffleNetV2Block(input_channel, output_channel, 2, 2))
                    else:
                        features.append(ShuffleNetV2Block(input_channel, output_channel, 1, 1))
                    input_channel = output_channel

                stage_name = 'stage_%d' % (idxstage + 1)
                self.stages.append(stage_name)
                self.add_module(stage_name, nn.Sequential(*features))
            else:
                output_channel = self.stage_out_channels[idxstage + 2]
                conv_last = conv_1x1_bn(input_channel, output_channel)
                # stage_name = "stage_%d" % (idxstage + 1)
                stage_name = "conv5"
                self.stages.append(stage_name)
                self.add_module(stage_name, conv_last)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, stage_name in enumerate(self.stages):
            feature_layer = self.__getattr__(stage_name)
            x = feature_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return tuple(outs)
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetV2, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            # frozen stem
            for m in self.conv1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                for params in m.parameters():
                    params.requires_grad = False
            # frozen stages
            for i in range(0, self.frozen_stages):
                mod = self.__getattr__(self.stages[i])
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

