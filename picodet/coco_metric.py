import torch
import os
import sys
import json
from pathlib import Path

from picodet.utils import get_categories
from picodet.coco_utils import cocoapi_eval


def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    # distribute flattened bboxes into sepcific image with help of bbox_nums
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

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        infer_res['bbox'] = get_det_res(outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    return infer_res


class COCOMetric:
    def __init__(self, anno_file,
                 classwise=False,
                 output_eval=None,
                 bias=0,
                 save_prediction_only=False):
        self.anno_file = anno_file
        self.clsid2catid, _ = get_categories(anno_file)
        self.classwise = classwise
        self.output_eval = output_eval
        self.bias = bias
        self.save_prediction_only = save_prediction_only

        if not self.save_prediction_only:
            assert os.path.isfile(anno_file), \
                    "anno_file {} not a file".format(anno_file)

        if self.output_eval is not None:
            Path(self.output_eval).mkdir(exist_ok=True)

        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        images, targets = inputs
        im_id = [t['im_id'] for t in targets]
        im_id = torch.stack(im_id, dim=0).cpu()
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    torch.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []


    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                print('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                print('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

    def get_results(self):
        return self.eval_results