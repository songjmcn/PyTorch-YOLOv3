import logging

import torch
import torch.nn as nn
from mmcv.cnn import ShuffleNetV2Block, ShuffleNetV2
from mmcv.cnn.weight_init import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES


from collections import namedtuple

ExtralSpec = namedtuple(
    "ExtralSpec",
    [
        "index",  # Index of the extral stage
        "stride",  # Stride
        "padding", # Padding
        "channels",  # Output channels
        'block_count', # Block count
        "return_features",  # True => return the last feature map from this stage
    ],
)


# ShuffleNetV2SSD300 = tuple(
#     ExtralSpec(index=i, stride=s, padding=p, channels=c,
#                block_count=b, return_features=r)
#     for (i, s, p, c, b, r) in ((5, 2, 1, 512, 1, True),
#                                (6, 2, 1, 256, 1, True),
#                                (7, 2, 1, 256, 1, True),
#                                (8, 2, 1, 256, 1, True))
# )

#parking
ShuffleNetV2SSD300 = tuple(
    ExtralSpec(index=i, stride=s, padding=p, channels=c,
               block_count=b, return_features=r)
    for (i, s, p, c, b, r) in ((5, 2, 1, 512, 1, True),)
)

# ShuffleNetV2SSD160 = tuple(
#     ExtralSpec(index=i, stride=s, padding=p, channels=c,
#                block_count=b, return_features=r)
#     for (i, s, p, c, b, r) in ()
# )

class ExtralStage(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 stride,
                 padding=0
                 ):
        super(ExtralStage, self).__init__()
        assert stride in [1, 2]

        oup_inc = inp // 2

        self.block = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(oup_inc, oup_inc, 3, stride, padding, groups=oup_inc,
                              bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(oup_inc, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

@BACKBONES.register_module
class ShuffleNetV2SSD(ShuffleNetV2):

    extra_setting = {
        300: ShuffleNetV2SSD300,
    }

    def __init__(self,
                 input_size,
                 num_stages=4,
                 out_indices=(1, 2, 3),
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 width_mult=1.0,
                 image_channel=3,
                 l2_norm_scale=20
                 ):
        """

        :param input_size:
        :param num_stages:
        :param out_indices:
        :param frozen_stages:  No Need for SSD
        :param bn_eval:  No Need for SSD
        :param bn_frozen:  No Need for SSD
        :param width_mult:
        :param l2_norm_scale:
        """
        super(ShuffleNetV2SSD, self).__init__(
            num_stages=num_stages,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            width_mult=width_mult,
            image_channel=image_channel,
        )

        assert input_size in (300, 160, 512)
        self.input_size = input_size

        # build extral stages
        self._build_extral_stages()
        # build norm layer
        #self.l2_norm = L2Norm(
        #    self.stage_out_channels[-1],
        #    l2_norm_scale)

    def _build_extral_stages(self):
        input_channel = self.stage_out_channels[-1]
        extral_specs = self.extra_setting[self.input_size]
        for extral_spec in extral_specs:
            output_channel = extral_spec.channels
            features = []
            num_repeat = extral_spec.block_count
            for i in range(num_repeat):
                if i == 0:
                    features.append(ExtralStage(input_channel, output_channel, 2, padding=1))
                else:
                    features.append(ExtralStage(input_channel, output_channel, 1))
                input_channel = output_channel

            stage_name = 'stage_%d' % (extral_spec.index)
            self.stages.append(stage_name)
            self.add_module(stage_name, nn.Sequential(*features))
            if extral_spec.return_features:
                self.out_indices.append(extral_spec.index - 1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("Body load checkpoint")
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        #constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, stage_name in enumerate(self.stages):
            feture_layer = self.__getattr__(stage_name)
            x = feture_layer(x)
            if i in self.out_indices:
                outs.append(x)
        #outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return tuple(outs)
        else:
            return tuple(outs)


class L2Norm(nn.Module):
    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight[None, :, None, None].expand_as(x) * x / norm


if __name__ == '__main__':
    ssd_ShuffleNet = ShuffleNetV2SSD(160, num_stages=4, out_indices=(1, 2,3), image_channel=1)
    print(ssd_ShuffleNet)
