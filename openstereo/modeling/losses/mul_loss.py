import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss
import pdb


class Smooth_l1_Loss(BaseLoss):
    def __init__(self, reduction='mean', loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.reduction = reduction
        

    def forward(self, disp_ests, disp_gt, mask=None):
        loss = F.smooth_l1_loss(
            disp_ests[mask] if mask is not None else disp_ests,
            disp_gt[mask] if mask is not None else disp_gt,
            reduction=self.reduction
        )
        self.info.update({'loss': loss})
        return loss, self.info


class Weighted_Smooth_l1_Loss(BaseLoss):
    def __init__(self, weights, reduction='mean', loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.weights = weights
        self.reduction = reduction

        
        

    def forward(self, disp_ests, disp_gt, mask=None):
        pdb.set_trace()
        
        weights = self.weights
        loss = 0.
        for disp_est, weight in zip(disp_ests, weights):
            loss += weight * F.smooth_l1_loss(
                disp_est[mask] if mask is not None else disp_est,
                disp_gt[mask] if mask is not None else disp_gt,
                reduction=self.reduction
            )
        self.info.update({'loss': loss})
        
        
        return loss, self.info
    

# class CrossEntropy_Loss(BaseLoss):
#     def __init__(self, reduction='mean', loss_term_weight=1.0):
#         super().__init__(loss_term_weight)
#         self.reduction = reduction

#     def forward(self, logits, labels, mask=None):
#         if mask is not None:
#             logits = logits[mask]
#             labels = labels[mask]
#         loss = F.cross_entropy(logits, labels, reduction=self.reduction)
#         self.info.update({'loss': loss})
#         return loss, self.info


class CE_Loss(BaseLoss):
    
    def __init__(self, reduction='mean', loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.reduction = reduction
    def forward(self, seg_ests, seg_gt, num_classes=19):
        """
        计算交叉熵损失
        
        参数:
        - inputs: 张量，形状为 [B, C, H, W]，表示模型的输出
        - target: 张量，形状为 [B, C, H, W]，表示onenehot编码的标签
        - num_classes: 类别的个数
        
        返回:
        - 交叉熵损失
        """
        # seg_ests = seg_ests.permute(0, 3,1, 2)
        # seg_gt = seg_gt.permute(0, 3,1, 2)
        # 检查输入张量的形状
        # pdb.set_trace()
        
        assert seg_ests.shape[1] == num_classes, "输入张量的通道数应等于类别的个数"
        seg_gt = torch.argmax(seg_gt, dim=1)
        seg_ests = seg_ests.permute(0, 2, 3, 1).reshape(-1, num_classes)
        seg_gt = seg_gt.view(-1)
        
        # Calculate the cross entropy loss
        loss = F.cross_entropy(seg_ests, seg_gt, reduction=self.reduction)
        



        
        # # # 将输入张量从 [B, C, H, W] 转换为 [B, C, H*W]
        # seg_ests = seg_ests.reshape(seg_ests.size(0), seg_ests.size(1), -1) #torch.Size([1, 35, 465750])
        

        # # # 将目标张量从 one-hot 编码转换为类别索引
        # seg_gt = torch.argmax(seg_gt, dim=1)  # 转换为形状 [B, H, W]  torch.Size([1, 375, 1242])
        
        
        # # # 将输入张量从 [B, C, H*W] 转换为 [B*H*W, C]
        # # pdb.set_trace()
        # seg_ests = seg_ests.permute(0, 2, 1).contiguous().reshape(-1, num_classes)
        
        
        
        # # # 将目标张量从 [B, H, W] 转换为 [B*H*W]
        # seg_gt = seg_gt.reshape(-1)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(seg_ests, seg_gt)
        # pdb.set_trace()
        self.info.update({'loss': loss})

        
        
        return loss, self.info

class CombinedLoss(BaseLoss):
    def __init__(self, weight_smooth_l1=0, weight_ce=0, reduction='mean'):
        super(CombinedLoss, self).__init__()
        # Initialize individual loss components
        self.smooth_l1_loss = Smooth_l1_Loss(reduction=reduction)
        self.cross_entropy_loss = CrossEntropy_Loss(reduction=reduction)
        # Weights for combining losses
        self.weight_smooth_l1 = weight_smooth_l1
        self.weight_ce = weight_ce

    def forward(self, disp_ests, disp_gt, logits, labels, mask=None):
        # Compute Smooth L1 Loss
        smooth_l1_loss, info_l1 = self.smooth_l1_loss(disp_ests, disp_gt, mask)

        # Compute Cross-Entropy Loss
        cross_entropy_loss, info_ce = self.cross_entropy_loss(logits, labels, mask)

        # Combine losses with respective weights
        combined_loss = self.weight_smooth_l1 * smooth_l1_loss + self.weight_ce * cross_entropy_loss

        # Update info dictionary with both loss components
        combined_info = {
            'smooth_l1_loss': info_l1['loss'],
            'cross_entropy_loss': info_ce['loss'],
            'combined_loss': combined_loss
        }
        self.info.update({'combined_loss': combined_loss})

        return combined_loss, combined_info


