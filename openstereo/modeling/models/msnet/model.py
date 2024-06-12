import torch
import matplotlib.pyplot as plt
from modeling.base_model import BaseModel
from modeling.base_mode_mul import BaseModel_mul
from .MSNet2D import MSNet2D
from .MSNet3D import MSNet3D
from .MSNet2D_mul import MSNet2D_mul
import pdb
import numpy as np






class MSNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self):
        if self.model_cfg['model_type'] == '3D':
            self.net = MSNet3D(self.max_disp)
        elif self.model_cfg['model_type'] == '2D':
            self.net = MSNet2D(self.max_disp)
        else:
            raise NotImplementedError

    def init_parameters(self):
        return

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)
        # pdb.set_trace()
        if self.training:
            pred0, pred1, pred2, pred3 = res
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": [pred0, pred1, pred2, pred3],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
                },
            }
           
        else:
            disp_pred = res[0]
            output = {
                "inference_disp": {
                    "disp_est": disp_pred,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': disp_pred[0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0]], dim=0),
                }
        return output


class MSNet_mul(BaseModel_mul):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        

    def build_network(self):
        if self.model_cfg['model_type'] in ['mul', 'mul_only_disp']:
            # pdb.set_trace()
            self.net = MSNet2D_mul(self.max_disp, self.num_classes)
        else:
            raise NotImplementedError

    def init_parameters(self):
        return

    def forward(self, inputs):
        """Forward the network."""
        # print("building MSNet2D_mul... ")
        # print("MSNet_mul inputs: ", inputs)   # dict_keys(['ref_img', 'tgt_img', 'disp_gt', 'mask', 'seg_labels', 'index'])
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
       

        # pdb.set_trace()
        # 输入左右视图
        res, seg_map = self.net(ref_img, tgt_img)
       
        if self.training:
            pred0, pred1, pred2, pred3 = res
            #如果inputs中有seg_labels
            if 'seg_labels' in inputs:
                output = {
                    "training_disp": {
                        "disp": {
                            "disp_ests": [pred0, pred1, pred2, pred3],
                            "disp_gt": inputs['disp_gt'],
                            "mask": inputs['mask']
                        },
                    },
                    "visual_summary": {
                        'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                        'image/train/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
                    },
                    "training_seg": {
                            "seg": {
                                "seg_ests": seg_map,
                                "seg_gt":inputs['seg_labels'],
                            },
                        },
                }
            else:
                output = {
                    "training_disp": {
                        "disp": {
                            "disp_ests": [pred0, pred1, pred2, pred3],
                            "disp_gt": inputs['disp_gt'],
                            "mask": inputs['mask']
                        },
                    },
                    "visual_summary": {
                        'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                        'image/train/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
                    },
                }

        else:
            disp_pred = res[0]
            # pdb.set_trace()
            if 'seg_labels' in inputs:
                output = {
                    "inference_disp": {
                        "disp_est": disp_pred,
                    },
                    "visual_summary": {
                        'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                        'image/test/disp_c': disp_pred[0],
                    },
                    "val_seg": {
                        
                        "seg_ests": seg_map,
                        "seg_gt": inputs['seg_labels'],
                        
                    }
                }
            else:
                output = {
                    "inference_disp": {
                        "disp_est": disp_pred,
                    },
                    "visual_summary": {
                        'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                        'image/test/disp_c': disp_pred[0],
                    }
                }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0]], dim=0),
                }
        return output
    

    def freeze_cost_volume_and_after(self):
        self.net.freeze_cost_volume_and_after()
    


    def freeze_backbone(self):
        self.net.freeze_backbone()