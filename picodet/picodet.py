import torch
import torch.nn as nn

class PicoDet(nn.Module):
    """
    Generalized Focal Loss network, see arxiv.org/abs/2006.04388

    Args:
        backbone(nn.Module): backbone cnn, namely Esnet
        neck(nn.Module): fnn cnn, namely csp-pan
        head(nn.Module): head cnn, namely picohead
    """

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        # prepare some paras deciding exportation options
        self.export_nms = True # 不用，暂时post-process必须nms
        self.export_post_process = True
        self.inputs = {}

    def _forward(self):
        backbone_feats = self.backbone(self.inputs)
        #print('backbone_feats[2]:')
        #print(backbone_feats[2])
        fpn_feats = self.neck(backbone_feats)
        #print('fpn_feats[3]:')
        #print(fpn_feats[3])
        cls_logits_list, bboxes_reg_list = self.head(fpn_feats, self.export_post_process)
        #print('bboxes_reg_list[1]')
        #print(bboxes_reg_list[1])
        # TODO: there is something wrong with bboxes head head
        if self.training or not self.export_post_process:
            return cls_logits_list, bboxes_reg_list
        else:
            #scale_factor = self.inputs['scale_factor']
            images, targets = self.inputs
            scale_factor = [t['scale_factor'] for t in targets]
            scale_factor = torch.stack(scale_factor, dim=0)
            # 暂时默认post_process都必须nms filter
            bboxes, bbox_num = self.head.post_process(
                [cls_logits_list, bboxes_reg_list], scale_factor, export_nms=True
            )
            #print('bboxes:')
            #print(bboxes)
            return bboxes, bbox_num

    def forward(self, inputs):
        self.inputs = inputs
        if self.training:
            loss = {}
            #under training mode, self._forward() outputs cls_logit_list and bboxes_reg_list
            cls_logits_list, bboxes_reg_list = self._forward()
            #generalized focal loss dictionary
            loss_gfl = self.head.get_loss([cls_logits_list, bboxes_reg_list],self.inputs)
            loss.update(loss_gfl)
            total_loss = torch.stack(list(loss.values())).sum()
            loss.update({'loss':total_loss})
            return loss
        else:
            # under eval mode, self._forward() outputs processed bboxes and bbox_num
            if not self.export_post_process:
                return {'picodet': self._forward()}
            else:
                bboxes, bbox_num = self._forward()
                output = {'bbox': bboxes, 'bbox_num': bbox_num}
                return output
