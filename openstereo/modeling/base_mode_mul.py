"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `openstereo/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_val(model)
"""
from abc import abstractmethod, ABCMeta

import torch
from torch import nn
import numpy as np
from base_trainer_mul import BaseTrainer_mul
from utils import get_msg_mgr, is_dict, get_attr_from, is_list, get_valid_args
from . import backbone as backbones
from . import cost_processor as cost_processors
from . import disp_processor as disp_processors
from .loss_aggregator_mul import LossAggregator_mul
import pdb

class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """

    @abstractmethod
    def build_network(self):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def build_loss_fn(self):
        """Build your optimizer here."""
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, inputs, device=None, **kwargs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs):
        """Forward the network."""
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, inputs):
        """Forward the network for one step."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute the loss."""
        raise NotImplementedError


class BaseModel_mul(MetaModel, nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BaseModel_mul, self).__init__()
        
        self.msg_mgr = get_msg_mgr()
        self.cfg = cfg
        self.model_cfg = cfg['model_cfg']
        self.model_name = self.model_cfg['model']
        self.max_disp = self.model_cfg['base_config']['max_disp']
        self.num_classes = self.model_cfg['base_config']['num_classes']


        self.msg_mgr.log_info(self.model_cfg)
        self.DispProcessor = None
        self.CostProcessor = None
        self.SegmentationProcessor = None   #seg
        self.Backbone = None
        self.loss_fn = None
        self.build_network()
        self.build_loss_fn()
        self.Trainer = BaseTrainer_mul
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

    def build_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg) for cfg in backbone_cfg])
            return Backbone
        raise ValueError("Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_cost_processor(self, cost_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(cost_processor_cfg):
            CostProcessor = get_attr_from([cost_processors], cost_processor_cfg['type'])
            valid_args = get_valid_args(CostProcessor, cost_processor_cfg, ['type'])
            return CostProcessor(**valid_args)
        if is_list(cost_processor_cfg):
            CostProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in cost_processor_cfg])
            return CostProcessor
        raise ValueError("Error type for -Cost-Processor-Cfg-, supported: (A list of) dict.")

    def build_disp_processor(self, disp_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(disp_processor_cfg):
            DispProcessor = get_attr_from([disp_processors], disp_processor_cfg['type'])
            valid_args = get_valid_args(DispProcessor, disp_processor_cfg, ['type'])
            return DispProcessor(**valid_args)
        if is_list(disp_processor_cfg):
            DispProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in disp_processor_cfg])
            return DispProcessor
        raise ValueError("Error type for -Disp-Processor-Cfg-, supported: (A list of) dict.")
    


    def build_network(self):
        model_cfg = self.model_cfg
        if 'backbone_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['backbone_cfg'])
            self.Backbone = self.build_backbone(cfg)
        if 'cost_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['cost_processor_cfg'])
            self.CostProcessor = self.build_cost_processor(cfg)
        if 'disp_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['disp_processor_cfg'])
            self.DispProcessor = self.build_disp_processor(cfg)

    def build_loss_fn(self):
        """Get the loss function."""
        loss_cfg = self.cfg['loss_cfg']
        # pdb.set_trace()
        self.loss_fn = LossAggregator_mul(loss_cfg)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)
        # 语义分割分支
        seg_out = self.SegmentationProcessor(inputs)
        inputs.update(seg_out)
        return {
        'disp_output': disp_out,  # 深度估计输出
        'seg_output': seg_out,    # 语义分割输出
        
    }

    def prepare_inputs(self, inputs, device=None, apply_max_disp=True, apply_occ_mask=False):
        """Reorganize input data for different models

        Args:
            inputs: the input data.
            device: the device to put the data.
            apply_max_disp: whether to apply max_disp to the mask.
            apply_occ_mask: whether to apply occ_mask to the mask.
        Returns:
            dict: training data including ref_img, tgt_img, disp image,
                  and other metadata.
        """
     
        processed_inputs = {
            'ref_img': inputs['left'],
            'tgt_img': inputs['right']
        }
        if 'disp' in inputs.keys():
            disp_gt = inputs['disp']
            # compute the mask of valid disp_gt
            mask = (disp_gt < self.max_disp) & (disp_gt > 0) if apply_max_disp else disp_gt > 0
            mask = mask & inputs['occ_mask'].to(torch.bool) if apply_occ_mask else mask
            processed_inputs.update({
                'disp_gt': disp_gt,
                'mask': mask,
            })

        if 'labels' in inputs.keys():
            
            labels_np = inputs['labels'].numpy()
            #转换为标签，one-hot
            # pdb.set_trace()
            labels_np=self.label_to_onehot(labels_np, self.colormap) 
            # pdb.set_trace()
            seg_gt = torch.from_numpy(labels_np)
            # pdb.set_trace()
            
            processed_inputs['seg_labels'] = seg_gt
            

        if 'occ_mask' in inputs.keys():
            processed_inputs['occ_mask'] = inputs['occ_mask']
        if 'occ_mask_right' in inputs.keys():
            processed_inputs['occ_mask_right'] = inputs['occ_mask_right']
        
        if not self.training:
            for k in ['pad', 'name']:
                if k in inputs.keys():
                    processed_inputs[k] = inputs[k]
        if device is not None:
            # move data to device
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device) if torch.is_tensor(v) else v
        processed_inputs['index'] = inputs['index']
        # pdb.set_trace()

        return processed_inputs
        

    def forward_step(self, batch_data, device=None):
        batch_inputs = self.prepare_inputs(batch_data, device)
        print("batch_inputs:",batch_inputs)
        outputs = self.forward(batch_inputs)
        training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']
        segmentation_output = outputs['segmentation_output']  # 语义分割输出
        return training_disp, segmentation_output, visual_summary

    def compute_loss(self, training_disp, inputs=None):
        """Compute the loss."""
        loss, loss_info = self.loss_fn(training_disp)
        return loss, loss_info
    


    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
    
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
            # Add the class map to the semantic map
            semantic_map[..., i] = class_map
        
        # Transpose to get the shape [B, K, H, W]
        
        semantic_map = np.transpose(semantic_map, (0, 3, 1, 2))
        
        
        return semantic_map
    


    def onehot_to_label(self, semantic_map, colormap):
        """
        Converts a one-hot encoded mask (B, H, W, K) to a label image (B, H, W, C)
        """
        # Find the index of the maximum value along the last dimension (K)
        x = np.argmax(semantic_map, axis=-1)
        
        # Convert the colormap to a numpy array
        colour_codes = np.array(colormap)
        
        # Initialize the label array
        B, H, W = x.shape
        C = colour_codes.shape[1] if len(colour_codes.shape) > 1 else 1
        label = np.zeros((B, H, W, C), dtype=np.uint8)
        
        # Map each index to the corresponding color in the colormap
        for b in range(B):
            label[b] = colour_codes[x[b]]
        
        return label
    
    

