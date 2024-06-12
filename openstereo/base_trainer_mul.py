import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn.functional as F
import itertools

from data.stereo_dataset_batch import StereoBatchDataset
from evaluation.evaluator import OpenStereoEvaluator
from modeling.common import ClipGrad, fix_bn
from utils import NoOp, get_attr_from, get_valid_args, mkdir
from utils.common import convert_state_dict
from utils.warmup import LinearWarmup
import pdb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class BaseTrainer_mul:
    def __init__(
            self,
            model: nn.Module = None,
            trainer_cfg: dict = None,
            data_cfg: dict = None,
            is_dist: bool = True,
            rank: int = None,
            # device: torch.device = torch.device('cpu'),
            device: torch.device = torch.device('cpu'),
            **kwargs
    ):
        
        self.model = model
        self.trainer_cfg = trainer_cfg
        self.data_cfg = data_cfg
        self.optimizer_cfg = trainer_cfg['optimizer_cfg']
        self.scheduler_cfg = trainer_cfg['scheduler_cfg']
        self.evaluator_cfg = trainer_cfg['evaluator_cfg']
        self.num_classes = trainer_cfg['num_classes']
        self.clip_grade_config = trainer_cfg.get('clip_grad_cfg', {})
        self.load_state_dict_strict = trainer_cfg.get('load_state_dict_strict', True)
        self.optimizer = None
        self.evaluator = NoOp()
        self.warmup_scheduler = NoOp()
        self.epoch_scheduler = NoOp()
        self.batch_scheduler = NoOp()
        self.clip_gard = NoOp()
        self.is_dist = is_dist
        self.rank = rank if is_dist else None
        self.device = torch.device('cuda', rank) if is_dist else device
        self.current_epoch = 0
        self.current_iter = 0
        self.save_path = os.path.join(
            self.trainer_cfg.get("save_path", ".././output"),
            self.data_cfg['name'], self.model.model_name, self.trainer_cfg['save_name']
        )
        # pdb.set_trace()
        self.msg_mgr = model.msg_mgr
        self.amp = self.trainer_cfg.get('amp', False)
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.build_model()
        self.build_data_loader()
        self.build_optimizer(self.optimizer_cfg)
        self.build_scheduler(self.scheduler_cfg)
        self.build_warmup_scheduler(self.scheduler_cfg)
        self.build_evaluator()
        self.build_clip_grad()

        self.colormap = [
            (0, 0, 0),
            (10, 0, 255), # sofa
            (0, 112,255), # wall
            (255, 0, 255), # table
            (0, 255, 163), # windows
            (224, 255, 8), # door
            (8, 255, 214), # ceiling
            (255, 194, 7), # floor
            (255, 0, 0), # refrigerator
            (140, 140, 140), # bed
            (255, 0, 20), # toilet
            (204, 70, 3), # chair
            (80, 50, 50), # bathtub
            (0, 255, 41), # sink
            (224, 5, 255), # blanket
            (160, 150, 20), # lamp
            (150, 5, 61), # book
            (255, 5, 153), # indoor-plant
            (0, 255, 61)
        ]

        self.class_names = [
            "other"
            "sofa",
            "wall", 
            "table",
            "windows",
            "door",
            "ceiling", 
            "floor",
            "refrigerator",
            "bed",
            "toilet",
            "chair",
            "bathtub",
            "sink",
            "blanket",
            "lamp",
            "book",
            "indoor-plant",
         ]

        self.best_loss = float('inf')
        self.current_loss = None  # 初始化 current_loss




    def build_model(self, *args, **kwargs):
        # apply fix batch norm
        if self.trainer_cfg.get('fix_bn', False):
            self.msg_mgr.log_info('fix batch norm')
            self.model = fix_bn(self.model)
        # init parameters
        if self.trainer_cfg.get('init_parameters', False):
            self.msg_mgr.log_info('init parameters')
            self.model.init_parameters()
        # for some models, we need to set static graph eg: STTR
        if self.is_dist and self.model.model_cfg.get('_set_static_graph', False):
            self.model._set_static_graph()

    def build_data_loader(self):
        self.msg_mgr.log_info(self.data_cfg)
        
        self.train_loader = self.get_data_loader(self.data_cfg, 'train')
        self.val_loader = self.get_data_loader(self.data_cfg, 'val')
        self.val_disp_loader = self.get_data_loader(self.data_cfg, 'val_disp')
        # pdb.set_trace()

    def get_data_loader(self, data_cfg, scope):
        # pdb.set_trace()
        dataset = StereoBatchDataset(data_cfg, scope)

        
        batch_size = data_cfg.get(f'{scope}_batch_size', 1)
        num_workers = data_cfg.get('num_workers', 4)
        pin_memory = data_cfg.get('pin_memory', False)
        shuffle = data_cfg.get(f'shuffle', False) if scope == 'train' else False
        if self.is_dist:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        sampler = BatchSampler(sampler, batch_size, drop_last=False)
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=dataset.collect_fn,
            num_workers=0,
            pin_memory=pin_memory,
        )
        return loader

    def build_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer

    def build_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        scheduler = get_attr_from([optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(scheduler, scheduler_cfg, ['scheduler', 'warmup', 'on_epoch'])
        scheduler = scheduler(self.optimizer, **valid_arg)
        if scheduler_cfg.get('on_epoch', True):
            self.epoch_scheduler = scheduler
        else:
            self.batch_scheduler = scheduler

    def build_warmup_scheduler(self, scheduler_cfg):
        warmup_cfg = scheduler_cfg.get('warmup', None)
        if warmup_cfg is None:
            return
        self.warmup_scheduler = LinearWarmup(
            self.optimizer,
            warmup_period=warmup_cfg.get('warmup_steps', 1),
            last_step=self.current_iter - 1,
        )

    def build_evaluator(self):
        
        metrics = self.evaluator_cfg.get('metrics', ['epe', 'd1_all', 'bad_1', 'bad_2', 'bad_3'])
        if 'val' in self.data_cfg.get('val_list', ''):   
            metrics.append('miou')
        # pdb.set_trace()
        self.evaluator = OpenStereoEvaluator(metrics)

    def build_clip_grad(self):
        clip_type = self.clip_grade_config.get('type', None)
        if clip_type is None:
            return
        clip_value = self.clip_grade_config.get('clip_value', 0.1)
        max_norm = self.clip_grade_config.get('max_norm', 35)
        norm_type = self.clip_grade_config.get('norm_type', 2)
        self.clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)

    def train_epoch(self):
        self.current_epoch += 1
        total_loss = 0.
        self.model.train()

        # # 在训练的前50个epoch中冻结cost volume后的权重
        # if self.current_epoch <= 100:
        #     self.model.freeze_backbone()
        # else:
        #     # 解冻所有权重
        #     for param in self.model.parameters():
        #         param.requires_grad = True
        
        #xiaofanweitiaojie backbon
        # self.model.freeze_backbone()


        # Freeze BN
        # pdb.set_trace()
        if self.trainer_cfg.get('fix_bn', False):
           
            self.model = fix_bn(self.model)
        self.msg_mgr.log_info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.train_loader)},"
            f" batch size: {self.train_loader.sampler.batch_size}"
        )
        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.train_loader), desc=f'Train epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        # for distributed sampler to shuffle data
        # the first sampler is batch sampler and the second is distributed sampler
        if self.is_dist:
            self.train_loader.sampler.sampler.set_epoch(self.current_epoch)
        for i, data in enumerate(self.train_loader):
            # for max iter training
            if self.current_iter > self.trainer_cfg.get('max_iter', 1e10):
                self.msg_mgr.log_info('Max iter reached.')
                break
            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                    # ISSUE:
                    #   1. use forward_step will cause torch failed to find unused parameters
                    #   this will cause the model can not sync properly in distributed training
                    batch_inputs = self.model.prepare_inputs(data, device=self.device)
                    # pdb.set_trace()
                    
                    outputs= self.model.forward(batch_inputs)
                    
                    # training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']
                    training_disp, visual_summary,training_seg = outputs['training_disp'], outputs['visual_summary'],outputs['training_seg']
                    loss_seg, loss_info = self.model.compute_loss(training_seg, inputs=batch_inputs)
                    loss_disp, loss_disp_info = self.model.compute_loss(training_disp, inputs=batch_inputs)
                    loss = loss_seg
                    self.current_loss = loss
                
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)  # optional
                self.clip_gard(self.model)
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
            else:

                batch_inputs = self.model.prepare_inputs(data, device=self.device)
                
                outputs = self.model.forward(batch_inputs)#training_disp, visual_summary
                # pdb.set_trace()
                if 'training_seg' in outputs:
                
                    training_disp, visual_summary,training_seg = outputs['training_disp'], outputs['visual_summary'],outputs['training_seg']
                else:
                    training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']

               
                # loss_seg, loss_info = self.model.compute_loss(training_seg, inputs=batch_inputs)

                loss_disp, loss_info = self.model.compute_loss(training_disp, inputs=batch_inputs)

                loss = loss_disp
                self.current_loss = loss
                loss.backward()
                
                self.clip_gard(self.model)
                self.optimizer.step()
            self.current_iter += 1
            with self.warmup_scheduler.dampening():
                self.batch_scheduler.step()
            total_loss += loss.item() if not torch.isnan(loss) else 0
            lr = self.optimizer.param_groups[0]['lr']
            log_iter = self.trainer_cfg.get('log_iter', 10)
            if i % log_iter == 0 and i != 0:
                pbar.update(log_iter) if i != 0 else pbar.update(0)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'epoch_loss': total_loss / (i + 1),
                    'lr': lr
                })
                loss_info.update(visual_summary)
            loss_info.update({'scalar/train/lr': lr})

            # pdb.set_trace()
            self.msg_mgr.train_step(loss_info)
            
        # update rest pbar
        rest_iters = len(self.train_loader) - pbar.n if self.is_dist and self.rank == 0 or not self.is_dist else 0
        pbar.update(rest_iters)
        pbar.close()
        total_loss = torch.tensor(total_loss, device=self.device)
        if self.is_dist:
            dist.barrier()
            # reduce loss from all devices
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= dist.get_world_size()
        with self.warmup_scheduler.dampening():
            self.epoch_scheduler.step()
        # clear cache
        if next(self.model.parameters()).is_cuda:
            torch.cuda.empty_cache()
        return total_loss.item() / len(self.train_loader)

    def train_model(self):
        self.msg_mgr.log_info('Training started.')
        total_epoch = self.trainer_cfg.get('total_epoch', 10)
        while self.current_epoch < total_epoch:
            self.train_epoch()
            if self.current_epoch % self.trainer_cfg['save_every'] == 0:
                self.save_ckpt()
            if self.current_epoch % self.trainer_cfg['val_every'] == 0:
                self.val_epoch()
            if self.current_iter >= self.trainer_cfg.get('max_iter', 1e10):
                self.save_ckpt()
                self.msg_mgr.log_info('Max iter reached. Training finished.')
                return
        self.msg_mgr.log_info('Training finished.')

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        # init evaluator
        apply_max_disp = self.evaluator_cfg.get('apply_max_disp', True)
        apply_occ_mask = self.evaluator_cfg.get('apply_occ_mask', False)

        # init metrics
        epoch_metrics = {}
        for k in self.evaluator.metrics:
            epoch_metrics[k] = {
                'keys': [],
                'values': [],
            }

        epoch_metrics_disp = {}
        for k in self.evaluator.metrics:
            epoch_metrics_disp[k] = {
                'keys': [],
                'values': [],
            }
        
        self.msg_mgr.log_info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.val_loader) + len(self.val_disp_loader)},"
            f" batch size: {self.val_loader.sampler.batch_size}"
        )

        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.val_loader), desc=f'Eval epoch {self.current_epoch}')
        else:
            pbar = NoOp()


        # Specify the directory to save the images
        save_dir = 'zimages_folder/infer_disp_16bit'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_seg_dir = 'zimages_folder/infer_seg'
        os.makedirs(save_seg_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_showdisp_dir = 'zimages_folder/color_disp'
        os.makedirs(save_showdisp_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_disp2_dir = 'zimages_folder/val2_disp'
        os.makedirs(save_disp2_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_disp2_16bit_dir = 'zimages_folder/val2_disp_16bit'
        os.makedirs(save_disp2_16bit_dir, exist_ok=True)  # Create the directory if it doesn't exist


        for i, data in enumerate(self.val_loader):
            # pdb.set_trace()
            batch_inputs = self.model.prepare_inputs(data, device=self.device, apply_max_disp=apply_max_disp,
                                                     apply_occ_mask=apply_occ_mask)

            with autocast(enabled=self.amp):
                out = self.model.forward(batch_inputs)
          
                inference_disp, visual_summary = out['inference_disp'], out['visual_summary']
 
                # 获取视差图
                # inference_disp_est = inference_disp['disp_est'].squeeze(0).cpu().numpy()
                # disp_gt = batch_inputs['disp_gt'].squeeze(0).cpu().numpy()
                 # 调用保存函数
                # self.save_disparity_images(disp_gt, inference_disp_est, save_showdisp_dir, i)
                # self.save_disparity_as_16bit_image(inference_disp_est, save_dir, i)
 
                if 'seg_labels' in batch_inputs:
                    
                    target = out["val_seg"]['seg_gt'].permute(0, 2, 3, 1)  # 真实标签
                    prediction = out["val_seg"]['seg_ests'].permute(0, 2, 1, 3)
                    val_data = {
                        'disp_est': inference_disp['disp_est'],
                        'disp_gt': batch_inputs['disp_gt'],
                        'mask': batch_inputs['mask'],
                        'seg_labels':target,
                        'seg_pred':prediction,
                    }
                    val_res = self.evaluator(val_data)
                    for k, v in val_res.items():
                        v = v.tolist() if isinstance(v, torch.Tensor) else v
                        
                        # pdb.set_trace()
                        epoch_metrics[k]['keys'].extend(batch_inputs['index'])
                        if not isinstance(v,list):
                            v = [v]
                        epoch_metrics[k]['values'].extend(v)
                    # 调用保存函数
                    seg_val_data = self.save_segmentation_images(batch_inputs, prediction, target, self.onehot_to_label, self.colormap, save_seg_dir, i)
                else:
                    pdb.set_trace()
                    val_data_disp = {
                        'disp_est': inference_disp['disp_est'],
                        'disp_gt': batch_inputs['disp_gt'],
                        'mask': batch_inputs['mask'],
                    }
                    val_res_disp = self.evaluator(val_data_disp)
                    for k, v in val_res_disp.items():
                        v = v.tolist() if isinstance(v, torch.Tensor) else v
                        epoch_metrics_disp[k]['keys'].extend(batch_inputs['index'])
                        if not isinstance(v, list):
                            v = [v]
                        epoch_metrics_disp[k]['values'].extend(v)
                # print("epoch_metrics",epoch_metrics)
                # pdb.set_trace()
                log_iter = self.trainer_cfg.get('log_iter', 10)
                if i % log_iter == 0 and i != 0:
                    pbar.update(log_iter)

                    if 'miou' in val_res:
                        pbar.set_postfix({
                            'epe': val_res['epe'].mean().item(),
                            'miou': val_res['miou'],
                            # 'mpa': mpa,
                        })
                    else:
                        pbar.set_postfix({
                            'epe_disp': val_res_disp['epe'].mean().item(),
                        })
        #测评第二个数据集
        if self.val_disp_loader:
            for i, data in enumerate(self.val_disp_loader):
                # pdb.set_trace()
                batch_inputs = self.model.prepare_inputs(data, device=self.device, apply_max_disp=apply_max_disp,
                                                        apply_occ_mask=apply_occ_mask)

                with autocast(enabled=self.amp):
                    out = self.model.forward(batch_inputs)
            
                    inference_disp, visual_summary = out['inference_disp'], out['visual_summary']
    
                    # 获取视差图
                    # inference_disp_est = inference_disp['disp_est'].squeeze(0).cpu().numpy()
                    # disp_gt = batch_inputs['disp_gt'].squeeze(0).cpu().numpy()
                    # 调用保存函数
                    # self.save_disparity_images(disp_gt, inference_disp_est, save_disp2_dir, i)
                    # self.save_disparity_as_16bit_image(inference_disp_est, save_disp2_16bit_dir, i)

                    val_data_disp = {
                        'disp_est': inference_disp['disp_est'],
                        'disp_gt': batch_inputs['disp_gt'],
                        'mask': batch_inputs['mask'],
                    }
                    val_res_disp = self.evaluator(val_data_disp)
                    for k, v in val_res_disp.items():
                        v = v.tolist() if isinstance(v, torch.Tensor) else v
                        epoch_metrics_disp[k]['keys'].extend(batch_inputs['index'])
                        if not isinstance(v, list):
                            v = [v]
                        epoch_metrics_disp[k]['values'].extend(v)

                    log_iter = self.trainer_cfg.get('log_iter', 10)
                    if i % log_iter == 0 and i != 0:
                        pbar.update(log_iter)

                        pbar.set_postfix({
                            'epe_disp': val_res_disp['epe'].mean().item(),
                        })


        # log to tensorboard
        self.msg_mgr.write_to_tensorboard(visual_summary, self.current_epoch)

        # update rest pbar
        rest_iters = len(self.val_loader) - pbar.n if self.is_dist and self.rank == 0 or not self.is_dist else 0
        pbar.update(rest_iters)
        pbar.close()

        if self.is_dist:
            dist.barrier()
            self.msg_mgr.log_debug("Start reduce metrics.")
            for metric, data in epoch_metrics.items():
                keys = torch.tensor(data["keys"]).to(self.device)
                values = torch.tensor(data["values"]).to(self.device)

                # Create tensors to store gathered data
                gathered_keys = [torch.zeros_like(keys) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]

                if dist.get_rank() == 0:
                    # Gather the keys and values from all devices to the master device
                    dist.gather(keys, gather_list=gathered_keys, dst=0)
                    dist.gather(values, gather_list=gathered_values, dst=0)
                else:
                    dist.gather(keys, dst=0)
                    dist.gather(values, dst=0)

                if dist.get_rank() == 0:
                    # Concatenate the gathered keys and values
                    concatenated_keys = torch.cat(gathered_keys, dim=0)
                    concatenated_values = torch.cat(gathered_values, dim=0)

                    # Create a dictionary to store the unique keys and their corresponding values
                    unique_dict = {}
                    for key, value in zip(concatenated_keys.tolist(), concatenated_values.tolist()):
                        if key not in unique_dict:
                            unique_dict[key] = value

                    # Update the keys and values in epoch_metrics
                    epoch_metrics[metric]["values"] = list(unique_dict.values())
                    epoch_metrics[metric]["keys"] = list(unique_dict.keys())

        if not self.is_dist or dist.get_rank() == 0:
            for metric in epoch_metrics:
                epoch_metrics[metric]["result"] = torch.mean(torch.tensor(epoch_metrics[metric]["values"])).item()
            visual_info = {}
            for k in epoch_metrics:
                visual_info[f'scalar/val/{k}'] = epoch_metrics[k]['result']
            self.msg_mgr.write_to_tensorboard(visual_info, self.current_epoch)
            metric_info = {k: v['result'] for k, v in epoch_metrics.items()}
            self.msg_mgr.log_info(f"Epoch {self.current_epoch} metrics: {metric_info}")
            
            # Reduce and log metrics for epoch_metrics_disp
            if epoch_metrics_disp:
                for metric in epoch_metrics_disp:
                    epoch_metrics_disp[metric]["result"] = torch.mean(torch.tensor(epoch_metrics_disp[metric]["values"])).item()
                visual_info_disp = {}
                for k in epoch_metrics_disp:
                    visual_info_disp[f'scalar/val_disp/{k}'] = epoch_metrics_disp[k]['result']
                self.msg_mgr.write_to_tensorboard(visual_info_disp, self.current_epoch)
                metric_info_disp = {k: v['result'] for k, v in epoch_metrics_disp.items()}
                self.msg_mgr.log_info(f"Epoch {self.current_epoch} disp metrics: {metric_info_disp}")

           
            # clear cache
            if next(self.model.parameters()).is_cuda:
                torch.cuda.empty_cache()
            # return epoch_metrics
            return {**epoch_metrics, **epoch_metrics_disp}

    @torch.no_grad()
    def test_kitti(self):
        self.model.eval()
        model_name = self.model.model_name
        data_name = self.data_cfg['name']
        output_dir = os.path.join(
            self.trainer_cfg.get("save_path", ".././output"),
            f"{data_name}/{model_name}/{data_name}_submit/disp_0"
        )
        os.makedirs(output_dir, exist_ok=True)
        self.msg_mgr.log_info("Start testing...")
        for i, inputs in enumerate(self.test_loader):
            ipts = self.model.prepare_inputs(inputs, device=self.device)
            with autocast(enabled=self.amp):
                output = self.model.forward(ipts)
            inference_disp, visual_summary = output['inference_disp'], output['visual_summary']
            disp_est = inference_disp['disp_est']
            # crop padding
            if 'pad' in ipts:
                pad_top, pad_right, _, _ = ipts['pad']
                # tensor[:0] is equivalent to remove this dimension
                if pad_right == 0:
                    disp_est = disp_est[:, pad_top:, :]
                else:
                    disp_est = disp_est[:, pad_top:, :-pad_right]
            # save to file
            img = disp_est.squeeze(0).cpu().numpy()
            img = (img * 256).astype('uint16')
            img = Image.fromarray(img)
            name = inputs['name']
            img.save(os.path.join(output_dir, name))
        self.msg_mgr.log_info("Testing finished.")

   

    def save_ckpt(self):
        # Only save model from master process
        if not self.is_dist or self.rank == 0:
            mkdir(os.path.join(self.save_path, "checkpoints/"))
            save_name = self.trainer_cfg['save_name']
            state_dict = {
                'model': self.model.state_dict(),
                'epoch': self.current_epoch,
                'iter': self.current_iter,
            }
            # for amp
            if self.amp:
                state_dict['scaler'] = self.scaler.state_dict()
            if not isinstance(self.optimizer, NoOp):
                state_dict['optimizer'] = self.optimizer.state_dict()
            if not isinstance(self.batch_scheduler, NoOp):
                self.msg_mgr.log_debug('Batch scheduler saved.')
                state_dict['batch_scheduler'] = self.batch_scheduler.state_dict()
            if not isinstance(self.epoch_scheduler, NoOp):
                self.msg_mgr.log_debug('Epoch scheduler saved.')
                state_dict['epoch_scheduler'] = self.epoch_scheduler.state_dict()

            # Save the model from the current epoch
            latest_save_name = os.path.join(self.save_path, "checkpoints/", f'{save_name}_latest.pt')
            torch.save(state_dict, latest_save_name)
            self.msg_mgr.log_info(f'Model saved to {latest_save_name}')

            # Save the best model if the current loss is the lowest
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                best_save_name = os.path.join(self.save_path, "checkpoints/", f'{save_name}_best.pt')
                torch.save(state_dict, best_save_name)
                self.msg_mgr.log_info(f'Best model saved to {best_save_name}')

        if self.is_dist:
            # for distributed training, wait for all processes to finish saving
            dist.barrier()


    def load_ckpt(self, path):
        if not os.path.exists(path):
            self.msg_mgr.log_warning(f"Checkpoint {path} not found.")
            return
        map_location = {'cuda:0': f'cuda:{self.rank}'} if self.is_dist else self.device
        checkpoint = torch.load(path, map_location=map_location)
        # pdb.set_trace()
        model_state_dict = convert_state_dict(checkpoint['model'], is_dist=self.is_dist)
        
        
        # self.model.load_state_dict(model_state_dict, strict=self.load_state_dict_strict)
        # 尝试加载模型权重并捕获未匹配键
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        # self.model.load_state_dict(model_state_dict, strict=False)
         # 打印未加载的权重
        if missing_keys:
            self.msg_mgr.log_warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            self.msg_mgr.log_warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")


        self.msg_mgr.log_info(f'Model loaded from {path}')
  
        # for amp
        if self.amp:
            if 'scaler' not in checkpoint:
                self.msg_mgr.log_warning('Loaded model is not amp compatible.')
            else:
                self.scaler.load_state_dict(checkpoint['scaler'])

        
        # 如果resume为False或strict为False，则跳过加载优化器和调度器
        if not self.trainer_cfg.get('resume', True) or not self.load_state_dict_strict:
            pdb.set_trace()
            return
        

        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
        self.msg_mgr.iteration = self.current_iter

        try:
            # load optimizer
            # if self.trainer_cfg.get('optimizer_reset', False):
            self.msg_mgr.log_info('Optimizer reset.')
            self.build_optimizer(self.optimizer_cfg)
            # else:
                # pdb.set_trace()

                # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load scheduler
            if self.trainer_cfg.get('scheduler_reset', False):
                self.msg_mgr.log_info('Scheduler reset.')
                self.build_scheduler(self.scheduler_cfg)
            else:
                if not isinstance(self.batch_scheduler, NoOp):
                    self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
                if not isinstance(self.epoch_scheduler, NoOp):
                    self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
            # load warmup scheduler
            if self.trainer_cfg.get('warmup_reset', False):
                self.msg_mgr.log_info('Warmup scheduler reset.')
                self.build_warmup_scheduler(self.scheduler_cfg)
            else:
                self.warmup_scheduler.last_step = self.current_iter - 1

        except KeyError:
            self.msg_mgr.log_warning('Optimizer and scheduler not loaded.')

        if not isinstance(self.warmup_scheduler, NoOp):
            self.warmup_scheduler.last_step = self.current_iter

    def resume_ckpt(self, restore_hint):
        restore_hint = str(restore_hint)
        if restore_hint == '0':
            return
        if restore_hint.isdigit() and int(restore_hint) > 0:
            save_name = self.trainer_cfg['save_name']
            save_name = os.path.join(
                self.save_path, "checkpoints/", f'{save_name}_epoch_{restore_hint:0>3}.pt'
            )
        else:
            save_name = restore_hint
        self.load_ckpt(save_name)



    #     return miou, mpa
    def overallAccuracy(self):
        # return all class overall pixel accuracy,AO评价指标
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
  
    def meanIntersectionOverUnion(self, confusionMatrix):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(confusionMatrix)
        union = np.sum(confusionMatrix, axis=0) + np.sum(confusionMatrix, axis=1) - np.diag(confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
    def precision(self,confusionMatrix):
        #precision = TP / TP + FP
        p = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
        return p
    
    def recall(self,confusionMatrix):
        #recall = TP / TP + FN
        r = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
        return r
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.num_classes)#过滤掉其它类别
        label = self.num_classes * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusionMatrix = count.reshape(self.num_classes, self.num_classes)
        return confusionMatrix
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.num_classes, self.num_classes))

    
    def label_to_onehot(self, labels, colormap):
        """
        Converts a segmentation label (B, H, W, C) to (B, K, H, W) where the second dim is a one
        hot encoding vector. K is the number of classes.
        """
        B, H, W, C = labels.shape
        K = len(colormap)
        
        # Initialize the semantic map
        semantic_map = np.zeros((B, H, W, K), dtype=np.float32)
        
        # Iterate over each color in the colormap
        for i, colour in enumerate(colormap):
            # Check equality for each label
            equality = np.equal(labels, colour)
            # Check equality across the color channels
            class_map = np.all(equality, axis=-1)
           # Only add to semantic map if class_map is not empty
            if np.any(class_map):
                semantic_map[..., i] = class_map
        
        # Transpose to get the shape [B, K, H, W]
        semantic_map = np.transpose(semantic_map, (0, 3, 1, 2))
        
        return semantic_map
    


    def onehot_to_label(self, semantic_map, colormap):
        """
        Converts a one-hot encoded mask (B, H, W, K) to a label image (B, H, W, C)
        """
        # Find the index of the maximum value along the last dimension (K)
        K = len(colormap)

        x = np.argmax(semantic_map, axis=-1)
        
        # Convert the colormap to a numpy array
        colour_codes = np.array(colormap)
        
        # Initialize the label array
        B, H, W = x.shape
        C = colour_codes.shape[1] if len(colour_codes.shape) > 1 else 1
        label = np.zeros((B, H, W, C), dtype=np.uint8)
        
        # Map each index to the corresponding color in the colormap
        # for b in range(B):
        #     label[b] = colour_codes[x[b]]
        # Map each index to the corresponding color in the colormap
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    idx = x[b, h, w]
                    if idx < K:  # Ensure the index is within the range of the colormap
                        label[b, h, w] = colour_codes[idx]
        
        return label
    
    # def compute_confusion_matrix(self,prediction, target,num_classes):
    #     pred_classes = torch.argmax(prediction, dim=-1).cpu().numpy().flatten()
    #     target_classes = torch.argmax(target, dim=-1).cpu().numpy().flatten()
    #     conf_matrix = confusion_matrix(target_classes, pred_classes, labels=np.arange(num_classes))
    #     return conf_matrix

    def plot_confusion_matrix(self, conf_matrix, class_names):
        fig, ax = plt.subplots(figsize=(30, 30))
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # 设置坐标轴刻度和标签
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

        # 在每个格子里添加文字
        thresh = conf_matrix.max() / 2.
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=15)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=600)
        plt.show()
        print('save confusion_matrix.png')


    def save_disparity_images(self, disp_gt, inference_disp_est, save_dir, index):
        """
        绘制并保存真实视差图和推理视差图。

        参数:
        - disp_gt: 真实视差图 (numpy 数组).
        - inference_disp_est: 推理的视差图 (numpy 数组).
        - save_dir: 图像保存目录.
        - index: 图像索引.

        返回:
        - None
        """
        # 绘制图像
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(disp_gt, cmap='jet')
        ax[0].set_title('Ground Truth Disparity')
        ax[0].axis('off')
        
        ax[1].imshow(inference_disp_est, cmap='jet')
        ax[1].set_title('Inference Disparity')
        ax[1].axis('off')

        # 保存图像
        plt.tight_layout()
        filename = os.path.join(save_dir, f'inference_disp_{index}.png')
        plt.savefig(filename)
        plt.close()
        print(f'Saved image: {filename}')  # Print the saved file path


    def save_disparity_as_16bit_image(self, inference_disp_est, save_dir, index):
        """
        绘制并保存真实视差图和推理视差图。

        参数:
        - disp_gt: 真实视差图 (numpy 数组).
        - inference_disp_est: 推理的视差图 (numpy 数组).
        - save_dir: 图像保存目录.
        - index: 图像索引.
        """
        disp_image = (inference_disp_est * 256).astype(np.uint16)
        filename = os.path.join(save_dir, f'inference_disp_{index}.png')
        image = Image.fromarray(disp_image, mode="I;16")
        image.save(filename, compression=None)
        print(f'Saved image: {filename}')

    

    def save_segmentation_images(self, batch_inputs, prediction, target, colormap_fn, colormap, save_dir, index):
        """
        绘制并保存语义分割的原始图像、重建的彩色图像和预测的彩色图像。

        参数:
        - batch_inputs: 包含原始图像的输入数据 (字典).
        - prediction: 模型预测的语义分割结果 (Tensor).
        - target: 真实标签 (Tensor).
        - colormap_fn: 颜色映射函数.
        - colormap: 颜色映射.
        - save_dir: 图像保存目录.
        - index: 图像索引.

        返回:
        - seg_val_data
        """
        # 将预测和目标转换为彩色图像
        prediction_cpu = prediction.cpu()
        target_cpu = target.cpu()
        seg_pred_color = colormap_fn(prediction_cpu, colormap)
        labels_tensor = colormap_fn(target_cpu, colormap)

        # 提取原始图像
        left_img = batch_inputs['ref_img'].cpu().squeeze(0).permute(1, 2, 0)
        seg_tensor = (left_img.numpy()).astype(np.uint8)
        seg_img = Image.fromarray(seg_tensor)

        # 可视化原始彩色图像和重建的彩色图像
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(seg_img)
        ax[0].set_title('Original Color Image')
        ax[1].imshow(labels_tensor.squeeze(0))
        ax[1].set_title('Reconstructed Color Image')
        ax[2].imshow(seg_pred_color.squeeze(0))
        ax[2].set_title('Predicted Color Image')

        # 保存图像
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'image_{index}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        print(f"Saved image {index} to {save_path}")

        seg_val_data={
                    'image/test/seg_pred':torch.from_numpy(seg_pred_color).permute(0, 3, 1, 2).to(torch.uint8),
                    'image/test/seg_gt':torch.from_numpy(labels_tensor).permute(0, 3, 1, 2).to(torch.uint8),
                }
        return seg_val_data




