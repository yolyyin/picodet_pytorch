import math
import weakref
import torch
import sys
sys.path.append('../picodet')

from picodet.esnet import ESNet
from picodet.csp_pan import CSPPAN
from picodet.pico_head import PicoHead, PicoFeat
from picodet.simota_assigner import SimOTAAssigner
from picodet.varifocal_loss import VarifocalLoss
from picodet.iou_loss import GIoULoss
from picodet.dfl_loss import DistributionFocalLoss
from picodet.utils import MultiClassNMS
from picodet.picodet import PicoDet
from torch.optim.lr_scheduler import LambdaLR


def create_model(num_classes=80):
    backbone = ESNet(scale=0.75, channel_ratio=[0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    neck = CSPPAN(in_channels=backbone._out_channels,
                  out_channels=96, use_depthwise=True,
                  num_csp_blocks=1, num_features=4)
    conv_feat = PicoFeat(feat_in=96, feat_out=96, num_convs=4, num_fpn_stride=4, norm_type='bn', share_cls_reg=True)
    loss_class = VarifocalLoss(use_sigmoid=True, iou_weighted=True)
    loss_dfl = DistributionFocalLoss(loss_weight=0.25)
    loss_box = GIoULoss(loss_weight=2.0)
    assigner = SimOTAAssigner(candidate_topk=10, iou_weight=6, num_classes=num_classes)
    nms = MultiClassNMS(nms_top_k=1000, keep_top_k=100, score_threshold=0., nms_threshold=0.6)
    head = PicoHead(conv_feat=conv_feat, num_classes=num_classes, fpn_stride=[8, 16, 32, 64], prior_prob=0.01,
                    loss_vfl=loss_class, loss_dfl=loss_dfl, loss_iou=loss_box, assigner=assigner,
                    reg_max=7, feat_in_chan=96, nms=nms, cell_offset=0.5)

    model = PicoDet(backbone, neck, head)
    return model


def create_optimizer(model, base_lr):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=base_lr,
                                momentum=0.9, weight_decay=4e-5)
    return optimizer


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, start_factor=0.1, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.start_factor = start_factor
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.start_factor*(1.-self.start_factor) * float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class ModelEMA(object):
    def __init__(self,
                 model,
                 decay = 0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1):
        self.step=0
        self.epoch=0
        self.decay=decay
        self.ema_decay_type=ema_decay_type
        self.cycle_epoch=cycle_epoch
        self.state_dict=dict()
        for k,v in model.state_dict().items():
            self.state_dict[k] = torch.zeros_like(v)
        self._model_state ={
            k:weakref.ref(p)
            for k,p in model.state_dict().items()
        }

    def reset(self):
        self.step=0
        self.epoch=0
        for k,v in self.state_dict.items():
            self.state_dict[k] = torch.zeros_like(v)

    def resume(self,state_dict,step=0):
        for k,v in state_dict.items():
            if k in self.state_dict:
                self.state_dict[k] = v.astype(self.state_dict[k].dtype)
        self.step = step

    def update(self, model=None):
        if self.ema_decay_type == 'threshold':
            decay = min(self.decay, float(1+self.step)/(10+self.step))
        elif self.ema_decay_type == 'exponential':
            decay = self.decay * (1 - math.exp(-float(self.step+1)/2000))
        else:
            decay = self.decay
        self._decay = decay

        if model is not None:
            model_dict = model.state_dict()
        else:
            # I don't know why this weak_ref thing does not work
            # by yty
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        for k, v in self.state_dict.items():
            v = decay * v + (1 - decay) * model_dict[k]
            v.stop_gradient = True
            self.state_dict[k] = v
        self.step += 1

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            if self.ema_decay_type != 'exponential':
                v = v / (1 - self._decay**self.step)
            v.stop_gradient = True
            state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict
