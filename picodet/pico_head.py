import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, GroupNorm
from picodet.utils import batch_distance2bbox, get_level_anchors, bbox2distance


class Integral(nn.Module):
    """
    一个用来计算离散分布期待值的module,
    在dfl loss算法中, 框的regulation logits被输出为4个离散分布
    框的预测值通过计算这4个离散分布的的期待值得到
    比如 top = sum(P(top_i)*top_i), 其中P(top_i)对应神经网络输出的logits矢量
    top_i是离散集{0,1,2,...,reg_max},
    代表在缩小的feature图上离anchor_points 0,1,..reg_max个像素，
    Args:
        reg_max(int):缩小的feature图上框边界离anchor_points距离的最大值
                     reg_max通常等于16,16是一个可更改经验值
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.possible_lengths = torch.linspace(0,self.reg_max,self.reg_max+1)

    def forward(self, x):
        """
        Args:
            x (Tensor): probability of bbox regression, shape (..., 4*(reg_max+1))
        Return:
            x (Tensor): Integral result of bbox regression, shape (..., 4)
        """
        x = F.softmax(x.reshape([-1,self.reg_max+1]), dim=1)
        values = self.possible_lengths.to(x.device)
        x = F.linear(x, values)
        if self.training:
            x = x.reshape([-1, 4])
        return x


class ConvNormLayer(nn.Module):
    """
    支持不同的norm类型, no act yet, by yty
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_groups=32):
        super(ConvNormLayer, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size-1)//2,
            groups=groups,
            bias=False)
        nn.init.normal_(self.conv.weight, mean=0., std=0.01)

        if norm_type in ['bn','sync_bn']:
            self.norm = BatchNorm2d(out_channels)
            # 既然coefficient都是0,此处L2Decay不管了, yty
            #weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            #bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        elif norm_type == 'gn':
            self.norm = GroupNorm(norm_groups, out_channels)
        else:
            self.norm = None

    def forward(self, inputs):
        y = self.conv(inputs)
        if self.norm is not None:
            y = self.norm(y)
        return y


class PicoFeat(nn.Module):
    """
    PicoFeat of PicoDet,
    一个dw,pw拆开的5*5 conv2d layer,不太明白为什么reg conv从cls conv的结果引出，
    以实际使用而言无妨,by yty

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 use_se=False):
        super(PicoFeat, self).__init__()
        if use_se:
            # sample code不支持se
            raise RuntimeError("SE is temporarily not supported in the codes! by yty")
        if not share_cls_reg:
            # sample code不支持分别卷积cls与reg
            raise RuntimeError("Temporarily only support share_cls_reg in the codes! by yty")
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.act = act
        self.cls_convs = []

        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                self.add_module(
                    'cls_conv_dw{}_{}'.format(stage_idx, i),
                    ConvNormLayer(
                        in_channels=in_c,
                        out_channels=feat_out,
                        kernel_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type))
                cls_conv_dw = self.get_submodule('cls_conv_dw{}_{}'.format(stage_idx, i))
                cls_subnet_convs.append(cls_conv_dw)
                self.add_module(
                    'cls_conv_pw{}_{}'.format(stage_idx, i),
                    # 总觉得此处的ch_in应该等于feat_out, 不知道是否写错了, by yty
                    ConvNormLayer(
                        in_channels=in_c,
                        out_channels=feat_out,
                        kernel_size=1,
                        stride=1,
                        groups=1,
                        norm_type=norm_type))
                cls_conv_pw = self.get_submodule('cls_conv_pw{}_{}'.format(stage_idx, i))
                cls_subnet_convs.append(cls_conv_pw)

            self.cls_convs.append(cls_subnet_convs)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat

        return cls_feat, reg_feat


class PicoHead(nn.Module):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_vfl (object): Instance of VariFocalLoss for class label loss
        loss_dfl (object): Instance of DistributionFocalLoss for bbox regulation loss
        loss_iou (object): Instance of IOU loss for bbox regulation loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7. QFL用到的积分变量, by yty
    """

    def __init__(self,
                 conv_feat,
                 loss_vfl,
                 loss_dfl,
                 loss_iou,
                 assigner,
                 num_classes=80,
                 fpn_stride=[8,16,32],
                 prior_prob=0.01,
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(PicoHead, self).__init__()
        self.conv_feat = conv_feat
        self.loss_vfl = loss_vfl
        self.loss_dfl = loss_dfl
        self.loss_iou = loss_iou
        self.assigner = assigner
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.distribution_project = Integral(self.reg_max)

        self.use_sigmoid = True #self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob)/self.prior_prob)

        self.head_cls_list = []
        for i in range(len(fpn_stride)):
            self.add_module(
                "head_cls" + str(i),
                Conv2d(in_channels=self.feat_in_chan,
                       out_channels=self.cls_out_channels + 4*(self.reg_max + 1),
                       kernel_size=1,
                       stride=1,
                       padding=0)
            )
            head_cls = self.get_submodule("head_cls" + str(i))
            torch.nn.init.normal_(head_cls.weight, mean=0., std=0.01)
            torch.nn.init.constant_(head_cls.bias, bias_init_value)
            self.head_cls_list.append(head_cls)

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(self.fpn_stride), "The size of fpn_feats is not equal to size of fpn_stride"

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            # head开始的conv,类似于frcnn里面的roi_pool, by yty
            conv_cls_feat, _ = self.conv_feat(fpn_feat, i)
            cls_logits = self.head_cls_list[i](conv_cls_feat)
            cls_score, bbox_pred = torch.split(
                cls_logits,
                [self.cls_out_channels, 4*(self.reg_max + 1)],
                dim=1,
            )

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)
            #print("cls_logits_list[0].shape:")
            #print(cls_logits_list[0].shape)
            #print("bboxes_reg_list[0].shape:")
            #print(bboxes_reg_list[0].shape)
            #print()

        return cls_logits_list, bboxes_reg_list

    def forward_eval(self, fpn_feats, export_post_process=True):
        anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        #print('anchor_points:')
        #print(anchor_points)
        #print('stride_tensor:')
        #print(stride_tensor)
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            # head开始的conv,类似于frcnn里面的roi_pool, by yty
            conv_cls_feat, _ = self.conv_feat(fpn_feat, i)
            #print('conv_cls_feat')
            #print(conv_cls_feat)
            cls_logits = self.head_cls_list[i](conv_cls_feat)
            #print('cls_logits')
            #print(cls_logits)
            cls_score, bbox_pred = torch.split(
                cls_logits,
                [self.cls_out_channels, 4 * (self.reg_max + 1)],
                dim=1,
            )
            #print('bbox_pred')
            #print(bbox_pred)
            #print()
            if not export_post_process:
                # Now only supports batch size = 1 in deploy
                # TODO(ygh): support batch size > 1
                cls_score_out = torch.sigmoid(cls_score).reshape(
                    [1, self.cls_out_channels, -1]).transpose([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4,
                                               -1]).transpose([0, 2, 1])
            else:
                _, _, h, w = fpn_feat.shape
                l = h * w
                cls_score_out = torch.sigmoid(
                    cls_score.reshape([-1, self.cls_out_channels, l]))
                # channel移动到最后, by yty
                bbox_pred = bbox_pred.permute(0, 2, 3, 1)
                #print('bbox_pred')
                #print(bbox_pred)
                bbox_pred = self.distribution_project(bbox_pred)
                #print('bbox_pred')
                #print(bbox_pred)
                bbox_pred = bbox_pred.reshape([-1, l, 4])
                #print('bbox_pred')
                #print(bbox_pred)

            cls_logits_list.append(cls_score_out)
            bboxes_reg_list.append(bbox_pred)

        if export_post_process:
            cls_logits_list = torch.cat(cls_logits_list, dim=-1)
            bboxes_reg_list = torch.cat(bboxes_reg_list, dim=1)
            #print('bboxes_reg_list:')
            #print(bboxes_reg_list)

            bboxes_reg_list = batch_distance2bbox(anchor_points,
                                                  bboxes_reg_list)
            #print('bboxes_reg_list:')
            #print(bboxes_reg_list)
            bboxes_reg_list *= stride_tensor

        return cls_logits_list, bboxes_reg_list

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            _, _, h, w = feats[i].shape
            device = feats[i].device
            shift_x = torch.arange(end=w, device=device) + self.cell_offset
            shift_y = torch.arange(end=h, device=device) + self.cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            #print('shift_y:')
            #print(shift_y)
            #print('shift_x:')
            #print(shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=torch.float32, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self, head_outs, scale_factor, export_nms=True):
        # pred_scores: shape [N,C,*], pred_bboxes: shape [N,*,4]
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            # rescale boxes to the original images' size
            scale_y, scale_x = torch.split(scale_factor, 1, dim=-1)
            scale_factor = torch.cat(
                [scale_x, scale_y, scale_x, scale_y],
                dim=-1).reshape((-1, 1, 4))
            pred_bboxes /= scale_factor
            #print('pred_bboxes:')
            #print(pred_bboxes)
            bbox_pred, bbox_num = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num

    def decode_and_flatten_preds(self,cls_scores,bbox_preds):
        """
        Decode and flatten predictions for training usage.
        Args:
            cls_scores: list of [N,C,H,W] class feature maps
            bbox_preds: list of [N,32,H,W] bbox feature maps
        Return:
            f_cls_preds: tensor with shape [N,*,C], flattened cls scores
            f_bboxes: tensor with shape [N,*,4], flattened and decoded x1y1x2y2 bboxes
            f_center_and_strides: tensor with shape [N,*,4], flattened centers and strides
            f_dfl_preds: tensor with shape [N,*,32], flattened corner preds for dfl loss
        """
        device = cls_scores[0].device
        n_imgs = cls_scores[0].shape[0]
        featmap_sizes = [(c.shape[-2], c.shape[-1])
                         for c in cls_scores]

        # 把预测到的ltrb模式bbox转换成x1y1x2y2模式
        decode_bboxes = []
        center_and_strides = []
        flatten_dfl_preds =[]
        for featmap_size, stride, bbox_pred in zip(featmap_sizes, self.fpn_stride, bbox_preds):
            # get single level anchor points and strides
            ys, xs = get_level_anchors(featmap_size, stride, device, self.cell_offset)

            strides = torch.ones_like(xs) * stride
            center_and_stride = torch.stack([xs, ys, strides, strides],
                                            -1).tile([n_imgs, 1, 1])  # [N,M,4]
            center_and_strides.append(center_and_stride)

            # decode bboxes
            center_in_feature = center_and_stride.reshape([-1, 4])[:, :-2] / stride  # [N*M,2]
            bbox_pred = bbox_pred.permute([0, 2, 3, 1]).reshape(
                [n_imgs, -1, 4 * (self.reg_max + 1)])  # [N,M,32]
            pred_distance = self.distribution_project(bbox_pred)  # [N*M,4]
            decode_bbox = batch_distance2bbox(center_in_feature, pred_distance).reshape(
                [n_imgs, -1, 4]) * stride  # [N,M,4]
            decode_bboxes.append(decode_bbox)
            flatten_dfl_preds.append(bbox_pred)

        flatten_cls_preds = [
            cls_pred.permute([0, 2, 3, 1]).reshape(
                [n_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]
        f_cls_preds = torch.cat(flatten_cls_preds, dim=1)  # [N,*,C]
        f_bboxes = torch.concat(decode_bboxes, dim=1)  # [N,*,4]
        f_center_and_strides = torch.concat(center_and_strides, dim=1)  # [N,*,4]
        f_dfl_preds = torch.cat(flatten_dfl_preds, dim=1)  # [N,*,32]
        return f_cls_preds, f_bboxes, f_center_and_strides, f_dfl_preds


    def get_loss(self,head_outs,gt_meta):
        _, targets = gt_meta
        #num_imgs = gt_meta['im_id'].shape[0]
        num_imgs = len(targets)
        # list(tensor([G,4])), list(tensor([G,1]))
        #gt_boxes, gt_labels = gt_meta['gt_bbox'], gt_meta['gt_class']
        gt_boxes = [t['gt_bbox'] for t in targets]
        gt_labels = [t['gt_class'] for t in targets]

        cls_scores, bbox_preds = head_outs  # [N,C,H,W] [N,32,H,W]
        # 每一层anchor points数量
        num_level_anchors = [
            c.shape[-2] * c.shape[-1] for c in cls_scores
        ]
        # 训练结果decode
        # [N,M,C], [N,M,4], [N,M,4], [N,M,32]
        f_cls_preds, f_bboxes, f_center_and_strides, f_dfl_preds = self.decode_and_flatten_preds(cls_scores,bbox_preds)
        total_num_anchors = f_cls_preds.shape[1]
        #print('f_cls_preds:')
        #print(f_cls_preds)
        #print('f_bboxes:')
        #print(f_bboxes)
        #print('f_center_and_strides:')
        #print(f_center_and_strides)

        # targets list and possitive sample number
        vfl_targets=[]  # class and objectiveless
        dfl_targets=[]  # fine regression
        iou_targets=[]  # iou regression
        fg_masks=[]  # possitive sample masks
        num_fg=0.0

        # SimOTA assigner决定每一张图片的正负样本
        for cls_pred, f_bbox_pred,center_and_stride,gt_box,gt_label\
            in zip(f_cls_preds.detach(),f_bboxes.detach(),
                   f_center_and_strides.detach(),gt_boxes,gt_labels):
            if gt_label.numel()==0:
                # empty gt
                vfl_target = cls_scores.new_zeros((0,self.num_classes))
                iou_target = cls_scores.new_zeros((0,4))
                dfl_target = cls_scores.new_zeros((0,))
                fg_mask = cls_scores.new_zeros(total_num_anchors).bool()
            else:
                (
                    gt_matched_classes,  # [n_pos,1]
                    fg_mask,  # [M]
                    pred_ious_this_matching,  # [pos]
                    matched_gt_inds,  # [pos]
                    num_fg_img,  # int
                ) = self.assigner(
                    torch.sigmoid(cls_pred), f_bbox_pred, center_and_stride,
                    gt_box, gt_label
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                pos_ins = torch.nonzero(fg_mask).squeeze(1)  # [pos]
                # vfl target
                vfl_target = torch.zeros_like(cls_pred)  #[M,C]
                vfl_target[pos_ins.long(), gt_matched_classes.squeeze(1).long()] = pred_ious_this_matching

                # ious target
                # Note that the gt_bbox here is in x1y1x2y2 format, this is a must for dataloader
                pos_gt_bbox = gt_box[matched_gt_inds]  # [pos,4]
                iou_target = pos_gt_bbox

                # regression target
                pos_center = center_and_stride[fg_mask, :-2]  # [pos,2]
                # divide by stride to return to the feature map grid size,
                # which is required by the dfl loss calculation
                strides = center_and_stride[fg_mask, 2:3]  # [pos,1]
                dfl_target = bbox2distance(pos_center/strides, pos_gt_bbox/strides,
                                           self.reg_max).reshape([-1])  # [pos*4]

            vfl_targets.append(vfl_target)
            dfl_targets.append(dfl_target)
            iou_targets.append(iou_target)
            fg_masks.append(fg_mask)

        vfl_targets = torch.cat(vfl_targets,0)  # [N*M,C]
        dfl_targets = torch.cat(dfl_targets,0)  # [all..pos*4]
        iou_targets = torch.cat(iou_targets,0)  # [all..pos,4]
        fg_masks = torch.cat(fg_masks,0)  # [N*M]


        # weight of bbox/dfl loss according to cls scores
        # same reasoning as the one for the positive samples in Varifocal loss
        weight_regression,_ = torch.sigmoid(f_cls_preds.detach()).reshape(
            [-1,self.cls_out_channels]).max(dim=-1)
        weight_regression = weight_regression[fg_masks]
        # [all..pos]


        # vfl pred
        vfl_preds = f_cls_preds.reshape([-1,self.cls_out_channels])  # [N*M,C]
        # iou pred
        iou_preds = f_bboxes.reshape([-1, 4])[fg_masks]  # [all..pos,4]

        # dfl pred
        # corner pred [N, 32, H, W]
        corner_pred = f_dfl_preds.reshape([-1, 4 * (self.reg_max + 1)])  # [N*M,32]
        pos_corner_pred = corner_pred[fg_masks]  # [all..pos,32]
        dfl_preds = pos_corner_pred.reshape([-1, self.reg_max+1])  # [all..pos*4,8]

        num_fg = max(num_fg,1)
        avg_factor = max(weight_regression.sum().item(),1)

        loss_vfl = self.loss_vfl(vfl_preds,vfl_targets).sum()/num_fg  # float
        loss_iou = self.loss_iou(iou_preds,iou_targets,weight_regression).sum()/avg_factor
        #loss_iou = self.loss_iou(iou_preds, iou_targets).sum() / num_fg
        # this extra divider 4 is used because dfl loss is applied individually
        # to the 4 sides of the bbox and summed together, a mean value is calculated
        loss_dfl = self.loss_dfl(dfl_preds,dfl_targets,
                                 weight_regression.repeat(4)).sum()/(4*avg_factor)
        #loss_dfl = self.loss_dfl(dfl_preds, dfl_targets).sum()/num_fg
        loss_states = dict(loss_vfl=loss_vfl, loss_iou=loss_iou, loss_dfl=loss_dfl)
        return loss_states




