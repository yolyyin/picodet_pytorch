import torch

class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        eps (float): epsilon to avoid divide by zero, default as 1e-10
    """

    def __init__(self, loss_weight=1., eps=1e-10):
        self.loss_weight = loss_weight
        self.eps=eps

    def __call__(self, pred_bbox, gt_bbox, weight=None):
        """
        Args:
            pred_bbox: tensor, shape [N,4]
            gt_bbox: tensor, shape [N,4]
        Return:
            loss: tensor, shape [N,]
        """
        if pred_bbox.shape[-1] !=4 or gt_bbox.shape[-1] != 4:
            raise IndexError

        tl = torch.maximum(pred_bbox[..., :2], gt_bbox[..., :2])  # [N,2]
        br = torch.minimum(pred_bbox[..., 2:], gt_bbox[..., 2:])  # [N,2]
        area_pred = torch.prod(pred_bbox[...,2:] - pred_bbox[...,:2], dim=-1)  # [N]
        area_gt = torch.prod(gt_bbox[...,2:] - gt_bbox[...,:2], dim=-1)  # [N]
        enclosed_tl = torch.minimum(pred_bbox[..., :2], gt_bbox[..., :2])  # [N,2]
        enclosed_br = torch.maximum(pred_bbox[..., 2:], gt_bbox[..., 2:])  # [N,2]

        is_overlapped = (tl < br).type(tl.dtype).prod(dim=-1)  # [N]
        overlap = torch.prod(br-tl, dim=-1) * is_overlapped  # [N]
        union = area_pred + area_gt - overlap + self.eps  # [N]
        ious = overlap/union  # [N]
        enclosed = torch.prod(enclosed_br-enclosed_tl, dim=-1) + self.eps  # [N]
        gious = ious - (enclosed-union)/enclosed  # [N]
        loss = (1 - gious) * self.loss_weight
        if weight is not None:
            loss = loss * weight
        return loss
