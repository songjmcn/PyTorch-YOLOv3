import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module

from torch.utils import model_zoo
from shufflenet_v2 import ShuffleNetV2
from utils.utils import build_targets, to_cpu, non_max_suppression
from utils.checkpoint import load_checkpoint
from utils.weight_init import constant_init, kaiming_init, normal_init
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            obj_mask=obj_mask.bool()
            noobj_mask=noobj_mask.bool()
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self,):
        super(EmptyLayer, self).__init__()
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x=F.upsample(x,scale_factor=self.scale_factor,mode=self.mode)
        return x
#yololayer=tuple(
#    ExtralSpec(anchor=a,num_class=n,img_dim=i)
#    for(a,n,i) in ([ (10,14), (23,27), (37,58)],80,256)
#)
class ShuffleNetV2YOlOv3(ShuffleNetV2):
    def __init__(self,
                 input_size,
                 num_class,
                 width_mult=1.,
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 image_channel=3,
                 l2_norm_scale=20
                ):
        """
        :param input_size:
        :param num_stages: default use 4
        :param out_indices:
        :param frozen_stages:  No Need for YOLO
        :param bn_eval:  No Need for YOLO
        :param bn_frozen:  No Need for YOLO
        :param width_mult:
        :param l2_norm_scale:
        """
        super(ShuffleNetV2YOlOv3, self).__init__(
            num_stages=4,
            out_indices=(1,2,3,),
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            width_mult=width_mult,
            image_channel=image_channel,
        )
        assert input_size in (224,256,300,352)
        self.input_size=input_size
        self.num_class=num_class
        self.yolo1_mask=3,4,5
        self.yolo1_anchors= [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
        self.yolo2_mask=1,2,3
        self.yolo2_anchors=[10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
        self._build_extral_layer()
    def _build_extral_layer(self):
        input_channel = self.stage_out_channels[-1]

        ###yolo1 layers
        conv1=nn.Conv2d(input_channel,256,1,1,0)
        bn1=nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)
        activation1=nn.LeakyReLU(0.1)
        conv2=nn.Conv2d(256,512,3,1,1)
        bn2=nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)
        activation2=nn.LeakyReLU(0.1)
        conv3=nn.Conv2d(512,(self.num_class+5)*3,1,1,0)
        bn3=nn.BatchNorm2d((self.num_class+5)*3, momentum=0.9, eps=1e-5)
        activation3=nn.LeakyReLU(0.1)

        anchor_idxs =self.yolo1_mask
        anchors = self.yolo1_anchors
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        yolo1=YOLOLayer(anchors,self.num_class,self.input_size)
        '''
        self.add_module("yolo1_conv1",conv1)
        self.add_module("yolo1_bn1",bn1)
        self.add_module("yolo1_leaky1",activation1)
        self.add_module("yolo1_conv2",conv2)
        self.add_module("yolo1_bn2",bn2)
        self.add_module("yolo1_leaky2",activation2)
        self.add_module("yolo1_conv3",conv3)
        self.add_module("yolo1_bn3",bn3)
        self.add_module("yolo1_leaky3",activation3)
        self.add_module("yolo1_detection",yolo1)
        '''
        self.add_module("yolo1_conv1",nn.Sequential(conv1,bn1,activation1))
        self.stages.append("yolo1_conv1")
        self.add_module("yolo1_conv2",nn.Sequential(conv2,bn2,activation2))
        self.stages.append("yolo1_conv2")
        self.add_module("yolo1_conv3",nn.Sequential(conv3,bn3,activation3))
        self.stages.append("yolo1_conv3")
        self.add_module("yolo1_detection",yolo1)
        self.stages.append("yolo1_detection")

        ###yolo2 layer2
        self.add_module("yolo2_route1",EmptyLayer())
        self.stages.append("yolo2_route1")
        self.add_module("yolo2_conv1",
        nn.Sequential(
            nn.Conv2d(256,128,1,1,0),
            nn.BatchNorm2d(256,momentum=0.9,eps=1e-5),
            nn.LeakyReLU(0.1)
        ))
        self.stages.append("yolo2_conv1")
        self.add_module("yolo2_upsample",Upsample(128,mode="nearest"))
        self.stages.append("yolo2_upsample")
        self.add_module("yolo2_route2",EmptyLayer())
        self.stages.append("yolo2_route2")
        self.add_module("yolo2_conv2",
        nn.Sequential(
            nn.Conv2d(384,256,3,1,1),
            nn.BatchNorm2d(384,momentum=0.9,eps=1e-5),
            nn.LeakyReLU(0.1)
        ))
        self.stages.append("yolo2_conv2")
        self.add_module("yolo2_conv3",
        nn.Sequential(
            nn.Conv2d(384,(self.num_class+5)*3,1,1,0)
        ))
        self.stages.append("yolo2_conv3")
        anchor_idxs =self.yolo2_mask
        anchors = self.yolo2_anchors
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        self.add_module("yolo2_detection",YOLOLayer(anchors,self.num_class,self.input_size))
        self.stages.append("yolo2_detection")
    def forward(self,x,targets=None):
        img_dim=x.shape[2]
        layer_outputs, yolo_outputs=[],[]
        loss=0
        x = self.conv1(x)
        x = self.maxpool(x)
        layer_outputs.append(x)
        for i,stage_name in enumerate(self.stages):
            if stage_name=="yolo2_route1":
                x=torch.cat((x),1)
            elif stage_name=="yolo2_route2":
                input1=layer_outputs[4]
                input2=x
                x=torch.cat((input1,input2),1)
            elif stage_name.find("detection")!=-1:
                feature_layer=self.__getattr__(stage_name)
                x,layer_loss=feature_layer(x,targets,img_dim)
                loss+=layer_loss
                yolo_outputs.append(x)
            else:
                feature_layer=self.__getattr__(stage_name)
                x=feature_layer(x)
            layer_outputs.append(x)
        yolo_outputs=to_cpu(torch.cat(yolo_outputs,1))
        return yolo_outputs if targets is None else (loss,yolo_outputs)
    def init_weights(self,pretrained=None):
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
if __name__=="__main__":
    yolo=ShuffleNetV2YOlOv3(256,3)
    print(yolo)
            




