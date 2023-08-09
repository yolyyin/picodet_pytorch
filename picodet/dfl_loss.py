import torch
import torch.nn as nn
import torch.nn.functional as F


def distribution_focal_loss(pred, target):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        target (Tensor): Target distance label for bounding boxes with
            shape (N,).
    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    dis_left = target.long()  # [N]
    dis_right = dis_left + 1  # [N]
    weight_left = dis_right.to(torch.float32) - target  # [N]
    weight_right = target - dis_left.to(torch.float32)  # [N]
    # [N]
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    """
    def __init__(self, loss_weight = 1.0):
        super(DistributionFocalLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight = None):
        """Forward function.
        Args:
            pred (Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None, with shape (N,).
        Returns:
            Tensor: Loss tensor with shape (N,).
        """
        loss = self.loss_weight * distribution_focal_loss(pred, target)  # [N]
        if weight is not None:
            loss = loss * weight  # [N]
        return loss
