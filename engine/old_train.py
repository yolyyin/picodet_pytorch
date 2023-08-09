import torch
import copy
import numpy as np
from tqdm.auto import tqdm
import time
from PIL import Image
from model import create_model, create_optimizer, WarmupCosineSchedule, ModelEMA
from dataset import *
from utils import save_loss_plot, save_model, resume_paras, resume_state
from picodet.utils import get_categories, get_infer_results, visualize_results
from picodet.coco_metric import COCOMetric
import torchvision.transforms.functional as F


class Trainer:
    def __init__(self, train_loader, eval_loader, model, output_dir,
                 eval_anno_dir, base_lr, max_epoch=300,warmup_step=300,
                 cycle_epoch=40, snapshot_epoch=10,
                 device=None, pre_weight_dir=None):
        self.train_loss_list=[]
        self.model = model
        self.eval_anno_dir = eval_anno_dir
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = output_dir
        self.device = device if device else torch.device('cuda')
        self.pre_weight_dir = pre_weight_dir
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.warmup_step = warmup_step
        self.cycle_epoch = cycle_epoch
        self.snapshot_epoch = snapshot_epoch

        # create optimizer
        self.opitimizer = create_optimizer(self.model,base_lr)
        # create scheduler
        step_per_epoch = len(train_loader)
        max_iter = max_epoch * step_per_epoch
        self.lr_scheduler = WarmupCosineSchedule(self.opitimizer,
                                                 warmup_step,max_iter)
        # create ema
        self.model_ema = ModelEMA(self.model, cycle_epoch=cycle_epoch)

        if self.pre_weight_dir:
            # resume pretrain weights
            if self.pre_weight_dir.rsplit('.',1)[-1] == 'pth':
                resume_state(self.model, self.opitimizer, pre_weight_dir)
            if self.pre_weight_dir.rsplit('.', 1)[-1] == 'state':
                resume_paras(self.model, pre_weight_dir)

        # TODO: create coco metric object
        self.coco_metric = COCOMetric(
            anno_file=eval_anno_dir,
            classwise=False,
            output_eval=f'{output_dir}',
            bias=0,
            save_prediction_only=False)

    def train(self, num_epochs):
        print('Training...')
        self.model.train()
        best_loss = float('inf')
        ema_best_loss = -float('inf')

        for epoch in range(num_epochs):
            s = time.time()
            print(f"\nEPOCH {epoch+1} of {num_epochs}")
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
            e = time.time()
            #print(f'prog_bar time: {e-s}')
            start = time.time()
            for i, data in enumerate(prog_bar):
            #for i, data in enumerate(self.train_loader):
                #print(data)
                s = time.time()
                # data to device
                images, targets = data
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
                inputs = (images, targets)
                """
                print("image dtype")
                print(images[0].dtype)
                print("target dtype")
                for v in targets[0].values():
                    if hasattr(v, 'dtype'):
                        print(v.dtype)
                """
                # model forward
                outputs = self.model(inputs)
                loss = outputs['loss']
                #print(loss)
                loss_value = loss.detach().cpu().item()

                self.train_loss_list.append(loss_value)

                # model backward
                loss.backward()
                self.opitimizer.step()
                self.lr_scheduler.step()
                self.opitimizer.zero_grad()
                self.model_ema.update(self.model)

                prog_bar.set_description(desc=f"Loss:{loss_value:.4f}")
                #print(f"Loss:{loss_value:.4f}")
                if loss_value < best_loss:
                    best_loss = loss_value
                    save_model('local_best_model', self.output_dir,
                               epoch, self.model, self.opitimizer)

                del data,inputs,loss
                torch.cuda.empty_cache()
                e = time.time()
                #print(f'data fetch-calculate time: {e - s}')

            end = time.time()
            print(f"Took {((end - start) / 60): .3f} minutes for epoch {epoch + 1}")


            # use ema to ensure the model is not trapped in the local optimum
            # and save ema_best
            is_snapshot = ((epoch + 1) % self.snapshot_epoch == 0) or (epoch == num_epochs - 1)
            if is_snapshot:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self.model_ema.apply())

                # eval and save
                bbox_eval_results = self.eval_for_model_ema()
                all_map = bbox_eval_results[0]
                if all_map > ema_best_loss:
                    ema_best_loss = all_map
                    save_model(f'best_model_e{epoch}', self.output_dir,
                                epoch, self.model, self.opitimizer)
                    save_loss_plot(f'loss_plot_e{epoch}',self.output_dir, self.train_loss_list)
                    print(f"Model ema best model is saved at epoch {epoch}.")

                # reset original weight
                self.model.load_state_dict(weight)
                self.model.train()

            save_loss_plot("loss_plot_final", self.output_dir, self.train_loss_list)
            save_model("model_final", self.output_dir, num_epochs,
                       self.model, self.opitimizer)

    def eval_for_model_ema(self):
        with torch.no_grad():
            print('Evaluating...')
            self.model.eval()
            program_bar = tqdm(self.eval_loader, total=len(self.eval_loader))
            start = time.time()

            for i, data in enumerate(program_bar):
                images, targets = data
                images = list(image.to(self.device) for image in images)
                targets = [{ k: v.to(self.device) for k, v in t.items() } for t in targets]
                inputs = (images, targets)
                outs = self.model(inputs)
                self.coco_metric.update(inputs, outs)
                del data, outs
                torch.cuda.empty_cache()

            end = time.time()
            print(f"Took {((end - start) / 60): .3f} minutes for evaluation")
            print("---------------------------------------------")
            self.coco_metric.accumulate()
            bbox_eval_results = self.coco_metric.get_results()['bbox']
            self.coco_metric.reset()
            return bbox_eval_results


    def eval(self):
        with torch.no_grad():
            self._eval()

    def _eval(self):
        print('Evaluating...')
        self.model.eval()
        program_bar = tqdm(self.eval_loader, total=len(self.eval_loader))
        start = time.time()

        for i, data in enumerate(program_bar):
            images, targets = data
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            inputs = (images, targets)
            outs = self.model(inputs)
            self.coco_metric.update(inputs, outs)
            del data, outs
            torch.cuda.empty_cache()


        end = time.time()
        print(f"Took {((end - start) / 60): .3f} minutes for evaluation")
        print("---------------------------------------------")
        self.coco_metric.accumulate()
        self.coco_metric.reset()

    def predict(self, image_paths, des_size=(288, 512), draw_thr=0.4,
                ):
        start = time.time()
        # preprocess
        clsid2catid, catid2name = get_categories(anno_file=self.eval_anno_dir)


        images = []
        targets =[]
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            # image_info
            ori_w, ori_h = image.size
            h_scale = float(des_size[0]) / ori_h
            w_scale = float(des_size[1]) / ori_w
            scale_factor = torch.tensor([h_scale, w_scale],
                                        dtype=torch.float32, device=self.device)
            im_id = torch.tensor([i],dtype=torch.int64, device=self.device)
            curr_iter = torch.tensor([i],dtype=torch.int64, device=self.device)
            im_shape = torch.tensor(list(des_size), dtype=torch.float32,
                                    device=self.device)
            transforms = create_val_transform(des_size)
            image = np.array(image)
            image = transforms(image=image,
                               bboxes=[], classes=[])['image']

            #image = F.resize(image, size=des_size)
            #image = F.to_tensor(image).to(torch.float32)
            images.append(image.to(self.device))

            target = {
                'im_id': im_id,
                'curr_iter': curr_iter,
                'im_shape': im_shape,
                'scale_factor': scale_factor
            }
            targets.append(target)
        # run infer
        self.model.eval()
        with torch.no_grad():
            outs = self.model((images, targets))

        # postprocess
        for key in ['im_shape', 'scale_factor', 'im_id']:
            data = [t[key] for t in targets]
            data = torch.stack(data, dim=0).cpu()
            outs[key] = data
        for key, value in outs.items():
            if hasattr(value,'numpy'):
                outs[key] = value.cpu().numpy()
        result = get_infer_results(outs, clsid2catid)
        # sort result for comparison convenience
        result = sorted(result, key=lambda k: (k['image_id'], k['category_id'], -k['score']))

        for i, image_path in enumerate(image_paths):
            path_wo_suffix = image_path.rsplit('., 1')[0]
            image = Image.open(image_path)
            image = visualize_results(image, result, im_id=i,
                                      catid2name=catid2name, threshold=draw_thr)
            image.save(f'{path_wo_suffix}_result.jpg')
        end = time.time()
        print(f'inference time used: {end - start} s.')
        torch.cuda.empty_cache()


def train():

    to_size = (288, 512)
    DEVICE = torch.device('cuda')
    # config
    output_dir = "D:/2023_det/output/0724"
    train_root = "D:/data/coco_simple/train2017"
    train_anno_dir = "D:/data/coco_simple/annotations/instances_train2017.json"
    val_root = "D:/data/coco_simple/val2017"
    val_anno_dir = "D:/data/coco_simple/annotations/instances_val2017.json"
    #pre_weight_dir = "D:/2023_det/output/baseline/baseline.state"
    pre_weight_dir = "D:/2023_det/output/0724/local_best_model.pth"
    infer_paths = ['D:/2023_det/inference/0.jpg',
                   'D:/2023_det/inference/1.jpg',
                   'D:/2023_det/inference/2.jpg']

    # create model
    model = create_model(num_classes=80).to(DEVICE)

    # create train transform
    train_transform = create_train_transform(to_size)
    # create eval transform
    val_transform = create_val_transform(to_size)

    # create train dataset and dataloader
    train_dataset = create_train_dataset(train_root, train_anno_dir, to_size, DEVICE,
                                         train_tf=train_transform)
    train_loader = create_train_loader(train_dataset, batch_size=80, num_workers=0)

    # create eval dataset and dataloader
    val_dataset = create_val_dataset(val_root, val_anno_dir, to_size, DEVICE,
                                     val_tf=val_transform)
    val_loader = create_val_loader(val_dataset, batch_size=32, num_workers=0)

    # create trainer, begin trainning
    trainer = Trainer(train_loader, val_loader, model,
                      output_dir, val_anno_dir, base_lr=0.01,
                      max_epoch=300,warmup_step=300,
                      cycle_epoch=40, snapshot_epoch=10,
                      device=DEVICE, pre_weight_dir=pre_weight_dir)

    #trainer.eval()
    #trainer.train(num_epochs=100)
    trainer.predict(infer_paths)



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    train()

