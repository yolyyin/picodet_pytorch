import torch
from picodet.utils import bboxes_iou
from picodet.varifocal_loss import varifocal_loss

class SimOTAAssigner(object):
    """Computes matching between predictions and ground truth.
        Args:
            center_radius (int | float, optional): Ground truth center size
                to judge whether a prior is in center. Default 2.5.
            candidate_topk (int, optional): The candidate top-k which used to
                get top-k ious to calculate dynamic-k. Default 10.
            iou_weight (int | float, optional): The scale factor for regression
                iou cost. Default 3.0.
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            num_classes (int): The num_classes of dataset.
            use_vfl (int): Whether to use varifocal_loss when calculating the cost matrix.
        """

    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 num_classes=80,
                 use_vfl=True):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl

    def get_geometry_constraint(self,
                                gt_bboxes,
                                center_and_strides):
        """
        Priori filter of anchor predictions based on centerness relation between
        gt_bboxes and anchor points. predictions surrounding the anchor points
        satisfying either one of the following conditions will be considered
        possible to be positive, and will be considered in dynamic k matching.
        condition 1. the anchor point is in any gt bbox
        condition 2. the anchor point is within a fixed range(center radius)
                     from any gt bbox center

        This priori filter is applied to reduce computation in the matching phase
        Args:
            gt_bboxes:tensor, shape [G,4], per image gt bboxes
            center_and_strides:tensor, shape [M,4], per image anchor points
                               and corresponding strides
        Return:
            fg_masks: tensor, shape [M], filter indicating which predictions are possibly positive
            geometry_relation: tensor, shape [G, n_valid_anchors], indicating which predictions
                               can waiver centerness punishment cost
        """
        n_anchors = center_and_strides.shape[0]
        # condition 1:
        # [G, M]
        gt_bboxes_l = gt_bboxes[:, 0:1].repeat(1, n_anchors)
        gt_bboxes_r = gt_bboxes[:, 2:3].repeat(1, n_anchors)
        gt_bboxes_t = gt_bboxes[:, 1:2].repeat(1, n_anchors)
        gt_bboxes_b = gt_bboxes[:, 3:4].repeat(1, n_anchors)
        b_l = center_and_strides[:, 0] - gt_bboxes_l
        b_r = gt_bboxes_r - center_and_strides[:, 0]
        b_t = center_and_strides[:, 1] - gt_bboxes_t
        b_b = gt_bboxes_b - center_and_strides[:, 1]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # [G, M, 4]
        in_box_matrix = bbox_deltas.min(dim=-1).values > 0.  # [G, M]
        in_box_anchor_filter = in_box_matrix.sum(dim=0) > 0  # [M]


        # condition 2
        gt_center_x = 0.5 * gt_bboxes[:, 0:1] + 0.5 * gt_bboxes[:, 2:3]
        gt_center_y = 0.5 * gt_bboxes[:, 1:2] + 0.5 * gt_bboxes[:, 3:4]
        # [G,M]
        gt_bboxes_l = gt_center_x.repeat(1, n_anchors) - center_and_strides[:, 2] * self.center_radius
        gt_bboxes_r = gt_center_x.repeat(1, n_anchors) + center_and_strides[:, 2] * self.center_radius
        gt_bboxes_t = gt_center_y.repeat(1, n_anchors) - center_and_strides[:, 3] * self.center_radius
        gt_bboxes_b = gt_center_y.repeat(1, n_anchors) + center_and_strides[:, 3] * self.center_radius
        c_l = center_and_strides[:, 0] - gt_bboxes_l
        c_r = gt_bboxes_r - center_and_strides[:, 0]
        c_t = center_and_strides[:, 1] - gt_bboxes_t
        c_b = gt_bboxes_b - center_and_strides[:, 1]
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)  # [G, M, 4]
        in_center_matrix = center_deltas.min(dim=-1).values > 0.  # [G, M]
        in_center_anchor_filter = in_center_matrix.sum(dim=0) > 0  # [M]

        # condition 1 or condition2 constructs fg_mask
        # [M]
        #fg_masks = in_box_anchor_filter | in_center_anchor_filter
        fg_masks = torch.logical_or(in_box_anchor_filter,in_center_anchor_filter)
        # geometry matrix for centerness cost calculation
        # [G, n_valid_anchors]
        #geometry_relation = in_box_matrix[:, fg_masks] & in_center_matrix[:, fg_masks]
        geometry_relation = torch.logical_and(in_box_matrix[:,fg_masks],
                                              in_center_matrix[:, fg_masks])

        return fg_masks, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        Args:
            cost: tensor(n_gt,n_valid)
            pair_wise_ious: tensor(n_gt,n_valid)
            gt_classes: tensor(n_gt,1)
            num_gt: int
            fg_mask: tensor(M)
        Return:
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(self.candidate_topk, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()  # pos number
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # [n_pos]
        gt_matched_classes = gt_classes[matched_gt_inds]  # [n_pos,1]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes  # [n_pos]
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def __call__(self,
                 f_clses,
                 f_bboxes,
                 f_center_and_stride,
                 gt_bboxes,
                 gt_labels):
        """
        Assign gt bboxes to predictions with SimOTA
        Args:
            f_clses: tensor, shape [*,C], per image bbox clses
            f_bboxes: tensor, shape [*,4], per image bbox regulations
            f_center_and_stride: tensor, shape [*,4], per image anchor points info
            gt_bboxes: tensor, shape [n_gt, 4], ground truth bboxes, in x1y1x2y2 format
            gt_labels: tensor, shape [n_gt, 1], ground truth class labels
        Returns:
            gt_matched_classes: tensor, shape [n_pos,1]
            fg_masks: tensor, shape [M]
            pred_ious_this_matching: tensor,shape [n_pos]
            matched_gt_inds: tensor, shape [n_pos]
            num_fg: int
        """
        n_gt = gt_bboxes.shape[0]
        n_pred = f_bboxes.shape[0]

        if n_gt == 0 or n_pred ==0:
            # no ground truth bbox, all the preds map to background
            # TODO: return negetive results
            gt_matched_classes = torch.empty([0,1],dtype = torch.int64)
            fg_masks = torch.zeros([n_pred])
            pred_ious_this_matching = torch.tensor([])
            matched_gt_inds = torch.tensor([])
            num_fg = 0
            return (gt_matched_classes,fg_masks,
                    pred_ious_this_matching,matched_gt_inds,num_fg)

        # [M], [n_gt,n_valid]
        fg_masks, geometry_relation = self.get_geometry_constraint(gt_bboxes,
                                                                   f_center_and_stride)
        #print('fg_masks:')
        #print(fg_masks)
        #print('geometry_relation:')
        #print(geometry_relation)

        f_bboxes = f_bboxes[fg_masks]  # [n_valid,4]
        f_clses = f_clses[fg_masks]  # [n_valid,C]

        n_valid = f_clses.shape[0]
        # gt_bboxes is in x1y1x2y2 format
        pair_wise_ious = bboxes_iou(gt_bboxes, f_bboxes, xyxy=True)  # [n_gt, n_valid]
        #print('pairwise ious:')
        #print(pair_wise_ious)
        """
        calculate loss matrix of shape [n_gt,n_valid], 
        loss for every valid anchor bbox to regress to a specific gt bbox
        """
        # iou_loss参考了yolox的做法
        # losses_iou = -torch.log(pair_wise_ious + 1e-8)
        # calculate giou loss
        losses_giou = bboxes_iou(gt_bboxes, f_bboxes, xyxy=True,giou=True)  # [n_gt, n_valid]
        #print('losses_giou:')
        #print(losses_giou)
        # change tensor shapes for class loss calculation
        # gt_label original shape is [n_gt,1]
        reshape_gt_labels = gt_labels.repeat([1, n_valid]).reshape([-1])  # [n_gt*n_valid]
        # calculate vfl class loss
        vfl_preds = f_clses.unsqueeze(0).repeat(
            [n_gt,1,1]).reshape([-1,self.num_classes])  # [n_gt*n_valid,C]
        vfl_targets = torch.zeros_like(vfl_preds)
        vfl_targets[torch.arange(vfl_targets.shape[0]),
                    reshape_gt_labels.long()] = pair_wise_ious.reshape([-1])  # [n_gt*n_valid,C]
        losses_vfl = varifocal_loss(
            vfl_preds,vfl_targets,
            use_sigmoid=False).sum(-1).reshape([n_gt, n_valid]) # [n_gt,n_valid]
        #print('losses_vfl:')
        #print(losses_vfl)
        # calculate cost matrix
        cost_matrix = (
            losses_vfl*self.cls_weight + losses_giou*self.iou_weight +
            (~geometry_relation).to(torch.float32) * float(1e8)
        )  # [n_gt,n_valid]
        #print('score:')
        #print(losses_vfl*self.cls_weight+losses_giou*self.iou_weight)
        #print('cost_matrix:')
        #print(cost_matrix)

        (
            num_fg,  # int
            gt_matched_classes,  # [n_pos,1]
            pred_ious_this_matching,  # [n_pos]
            matched_gt_inds,  # [n_pos]
        ) = self.simota_matching(cost_matrix, pair_wise_ious,
                                    gt_labels, n_gt, fg_masks)
        del losses_vfl, cost_matrix, pair_wise_ious, losses_giou

        return(
            gt_matched_classes,
            fg_masks,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )