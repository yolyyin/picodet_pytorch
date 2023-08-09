import torch
import torchvision.ops as ops
import os
from PIL import ImageDraw
from pycocotools.coco import COCO


def get_categories(anno_file=None):
    """
    Get class id to category id map and category id
    to category name from coco annotation file.
    Args:
        anno_file(str): annotation file path
    Return:
         clsid2catid(dict), catid2name(dict):
         map cls id in actual code to category id & name in coco
    """
    if anno_file == None or (not os.path.isfile(anno_file)):
        raise ValueError(
            "anno_file '{}' is None or not set or not exist, "
            "please recheck anno_path. ".
            format(anno_file)
        )

    coco = COCO(anno_file)
    cats = coco.loadCats(coco.getCatIds())
    clsid2catid = { i: cat['id'] for i, cat in enumerate(cats) }
    catid2name = { cat['id']: cat['name'] for cat in cats }
    return clsid2catid, catid2name


def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    bbox_res = get_det_res(outs['bbox'], outs['bbox_num'],
                           outs['im_id'], catid, bias=bias)
    return bbox_res


def visualize_results(image, bboxes, im_id, catid2name, threshold):
    """
    draw final bbox result on images
    """
    draw = ImageDraw.Draw(image)
    color = (255,0,127)
    for det in bboxes:
        if im_id != det['image_id']:
            continue
        catid, bbox, score = det['category_id'], det['bbox'], det['score']
        if score < threshold:
            continue
        # draw bbox
        xmin,ymin,w,h = bbox
        xmax = xmin+w
        ymax = ymin+h
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax),
             (xmax, ymin), (xmin, ymin)],
            width=2,
            fill=color
        )
        # draw label
        text = f"{catid2name[catid]} {score:.2f}"
        tw, th = draw.textsize(text)
        draw.rectangle([(xmin+1, ymin-th), (xmin+tw+1, ymin)], fill=color)
        draw.text((xmin+1, ymin-th), text, fill=(255,255,255))
    return image

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, giou=False):
    """
    Args:
        bboxes_a: tensor, shape [N,4]
        bboxes_b: tensor, shape [M,4]
        xyxy: bool, true if bboxes are in xyxy format,
              flase if bboxes are in xywh format
    Return:
        pairwise iou: tensor, shape [N,M]
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[None, :, :2])  # [N,M,2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[None, :, 2:])  # [N,M,2]
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  # [N]
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)  # [M]
        if giou:
            enclosed_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[None, :, :2])  # [N,M,2]
            enclosed_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[None, :, 2:])
    else:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[None, :, :2])
        br = torch.min(
            bboxes_a[:, None, :2] + bboxes_a[:, None, 2:],
            bboxes_b[None, :, :2] + bboxes_b[None, :, 2:]
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)  # [N]
        area_b = torch.prod(bboxes_b[:, 2:], 1)  # [M]
        if giou:
            enclosed_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[None, :, :2])
            enclosed_br = torch.max(
                bboxes_a[:, None, :2] + bboxes_a[:, None, 2:],
                bboxes_b[None, :, :2] + bboxes_b[None, :, 2:]
            )  # [N,M,2]


    is_overlapped = (tl < br).type(tl.dtype).prod(dim=2)  # [N,M]
    area_overlap = torch.prod(br-tl, 2) * is_overlapped  # [N,M]
    union = area_a[:, None] + area_b[None, :] - area_overlap
    ious = area_overlap/union
    if giou:
        enclosed_wh = (enclosed_br-enclosed_tl).clamp(min=0)
        enclosed_area = torch.maximum(torch.tensor([1e-6]).to(enclosed_wh.device),
                                      enclosed_wh[:,:,0] * enclosed_wh[:,:,1])
        gious = ious - (enclosed_area-union)/enclosed_area
        return 1 - gious
    else:
        return ious



def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2] or [N,2], "xy" format
        distance (Tensor): [B, ..., 4] or [N,4], "ltrb" format
        max_shapes (Tensor): [B, 2] or [2], "h,w" format, Shape of the image.
    Returns:
        Tensor: [B, ..., 4] or [N,4] Decoded bboxes, "x1y1x2y2" format.
    """
    # TODO: there are bugs with this function!!
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.dim() - max_shapes.dim()
        for _ in range(delta_dim):
            max_shapes.unsqueeze(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox,
                                torch.zeros_like(out_bbox))
    return out_bbox


def get_level_anchors(featmap_size, stride, device, cell_offset=0.5):
    """
    generate anchors according to feature map sizes and strides
    Args:
        featmap_size: tuple, (h,w)
        stride: float, the scaling factor to scale back feature maps
        cell_offset: float, the center offset of anchor points,
                     default 0.5
    Return:
         (y,x): single dimension tensor of anchor ys and anchor xs
    """
    h, w = featmap_size
    x_range = (torch.arange(w,device=device,dtype=torch.float32)+cell_offset)*stride
    y_range = (torch.arange(h, device=device, dtype=torch.float32) + cell_offset) * stride
    y, x = torch.meshgrid(y_range,x_range,indexing='ij')
    y = y.flatten()
    x = x.flatten()
    return y, x


def bbox2distance(points,bbox,max_dis=None,eps=0.1):
    """Decode bounding box based on distances.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    Returns:
        Tensor: Shape (n,4), Decoded distances.
    """
    left = points[:,0] - bbox[:,0]  # [n]
    top = points[:, 1] - bbox[:,1]
    right = bbox[:,2] - points[:,0]
    bottom = bbox[:,3] - points[:,1]
    if max_dis is not None:
        left = left.clamp(min=0, max = max_dis-eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left,top,right,bottom],-1)


def multiclass_nms(pred_bboxes,pred_clses,nms_top_k,keep_top_k,score_thr,nms_thr):
    """
    Args:
        pred_bboxes: tensor, shape [N,*,4]
        pred_clses: tensor, shape [N,C,*]
        nms_top_k: the maximum number of top score boxes entering nms
        keep_top_: the maximum number of top score boxes after nms
        score_thr: the score threshold of valid bboxes before sending to nms
        nms_thr: the nms iou threshold for canceling out overlapping bbox
    Return:
        output_preds: tensor, shape [*,6], all the output predictions
                      concatenated together
        output_bbox_num: tensor, shape [num_of_images], tensor indicating the
                         number of bboxes each image contains
    """
    num_cls = pred_clses.shape[1]
    device = pred_bboxes.device
    output_preds = []
    output_bbox_num = []
    for pred_bbox, pred_cls in zip(pred_bboxes,pred_clses):
        # shape of pred_cls: [C,*], shape of pred_bbox:[*,4]

        # filter through score_thr,
        max_scores, labels = torch.max(pred_cls, 0)
        score_thr_idx = torch.where(max_scores >= score_thr)[0]
        pred_cls = pred_cls[:, score_thr_idx]  # [C,*]
        scores = max_scores[score_thr_idx]  # [*]
        labels = labels[score_thr_idx]  # [*]
        pred_bbox = pred_bbox[score_thr_idx, :]  # [*,4]

        # filter through nms_top_k
        # 想了半天，连pytorch官方都循环了我就不挣扎了TT，见torch.batched_nms, yty
        valid_idx = []
        for class_id in range(num_cls):
            curr_indices = torch.where(labels == class_id)[0]  # [sc]
            single_cls_idx=torch.argsort(scores[curr_indices],descending=True)[:nms_top_k] # [nms_top_k]
            valid_idx.append(curr_indices[single_cls_idx]) # [<nms_top_k]
        valid_idx = torch.cat(valid_idx)
        scores = scores[valid_idx]
        labels = labels[valid_idx]
        pred_bbox = pred_bbox[valid_idx,:]

        # torch batched_nms
        keep = ops.batched_nms(pred_bbox,scores,labels,nms_thr)
        # filter through keep_top_k
        keep = keep[:keep_top_k]
        # [*,4] [*] [*]
        pred_bbox,scores,labels = pred_bbox[keep],scores[keep,None],labels[keep,None]

        # 转为一个[*,6]矩阵
        predictions = torch.cat([labels,scores,pred_bbox], dim=-1)
        output_preds.append(predictions)
        output_bbox_num.append(keep.shape[0])
    output_preds = torch.cat(output_preds,dim=0)
    output_bbox_num = torch.tensor(output_bbox_num,dtype=torch.int64,device=device)
    return output_preds, output_bbox_num


class MultiClassNMS(object):
    def __init__(self,
                 nms_top_k=1000,
                 keep_top_k=100,
                 score_threshold=.05,
                 nms_threshold=.5):
        super().__init__()
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, bboxes, scores):
        """
        Args
            bboxes: tensor, shape [N,M,4], M is the number of bboxes
            score: tensor, shape [N,C,M], C is the number of classes
        Return:
            output_preds: tensor, shape [*,6], all the output predictions
                      concatenated together
            output_bbox_num: tensor, shape [num_of_images], tensor indicating the
                         number of bboxes each image contains
        """
        return multiclass_nms(bboxes, scores,
                              self.nms_top_k, self.keep_top_k,
                              self.score_threshold, self.nms_threshold)


