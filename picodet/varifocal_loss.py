import torch
import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(preds,
                   targets,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   use_sigmoid=True):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        preds (Tensor): The prediction with shape (N, C), C is the
            number of classes
        targets (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
    Return:
        loss (Tensor): has the same shape with input preds/targets,
            no reduction is applied.
    """
    # check that targets and preds are of the same size
    assert preds.shape == targets.shape
    if use_sigmoid:
        preds_new = torch.sigmoid(preds)
    else:
        preds_new = preds
    targets = targets.to(preds.dtype)
    if iou_weighted:
        # vari_focal_weight is different for positive and negative labels
        # class loss for positive labels is weighted by the target iou(the 'accuracy level')
        # whereas for negative labels the misclassified samples are emphasized.
        focal_weight = targets * (targets > 0.0).to(torch.float32) + \
            alpha * preds_new.abs().pow(gamma) * \
            (targets <= 0.0).to(torch.float32)
    else:
        focal_weight = (targets > 0.0).to(torch.float32) + \
            alpha * preds_new.abs().pow(gamma) * \
            (targets <= 0.0).to(torch.float32)

    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            preds, targets, reduction='none') * focal_weight
    else:
        loss = F.binary_cross_entropy(
            preds, targets, reduction='none') * focal_weight
    return loss


class VarifocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid = True,
                 alpha = 0.75,
                 gamma = 2.0,
                 iou_weighted = True,
                 loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
        Returns:
            Tensor: The calculated loss, same shape with pred/target
        """
        loss = self.loss_weight * varifocal_loss(pred,
                              target,
                              alpha=self.alpha,
                              gamma=self.gamma,
                              iou_weighted=self.iou_weighted,
                              use_sigmoid=self.use_sigmoid)
        if weight is not None:
            loss = loss * weight
        return loss

