import math

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from .submodule import feature_extraction, MobileV2_Residual, convbn, interweave_tensors, disparity_regression, SegmentationDecoder, ASPP,CustomLayer,MaxPoolAndStack,SegBranch
import pdb

class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class MSNet2D_mul(nn.Module):
    def __init__(self, maxdisp, num_classes):

        super(MSNet2D_mul, self).__init__()

        self.num_classes = num_classes
        #gpt
        self.SegBranch=SegBranch(num_classes)

        self.maxdisp = maxdisp

        self.num_groups = 1

        self.volume_size = 48

        self.hg_size = 48

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(256, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)

        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        # 冻结语义分割分支
        # self.freeze_segmentation_branch()


         
        # 冻结 cost volume 和之后的网络结构
        # self.freeze_cost_volume_and_after()
        # 冻结除 SegBranch 之外的所有部分
        # self.freeze_except_SegBranch()
        #冻结除了cost volume 的网络结构
        self.freeze_except_cost_volume()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        
        features_L, L1,L2 = self.feature_extraction(L)
        features_R, _, _ = self.feature_extraction(R)

        featL = self.preconv11(features_L)
        
        featR = self.preconv11(features_R)
        

        segout = self.SegBranch(L1,L2,features_L,featL)
 
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = torch.unsqueeze(cost0, 1)# torch.Size([1, 1, 48, 64, 128])
            
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')#torch.Size([1, 34, 256, 512])
            
            cost0 = torch.squeeze(cost0, 1) #torch.Size([1, 192, 256, 512])
            
            pred0 = F.softmax(cost0, dim=1) #torch.Size([1, 192, 256, 512])
            
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
  
            return [pred0, pred1, pred2, pred3], segout

        else:
            cost3 = self.classif3(out3)
           
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)


            segout = torch.transpose(segout, 1, 3)
           
            return [pred3], segout
        

    def freeze_segmentation_branch(self):
        for param in self.aspp.parameters():
            param.requires_grad = False
        for param in self.segmentation_decoder.parameters():
            param.requires_grad = False


    def load_pretrained_model(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        # 过滤掉语义分割分支相关的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith(('aspp', 'segmentation_decoder'))}
        # 更新当前模型的参数
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        

    def freeze_cost_volume_and_after(self):
        # 冻结 cost volume 部分
        for param in self.conv3d.parameters():
            param.requires_grad = False
        for param in self.volume11.parameters():
            param.requires_grad = False
        for param in self.dres0.parameters():
            param.requires_grad = False
        for param in self.dres1.parameters():
            param.requires_grad = False
        for param in self.encoder_decoder1.parameters():
            param.requires_grad = False
        for param in self.encoder_decoder2.parameters():
            param.requires_grad = False
        for param in self.encoder_decoder3.parameters():
            param.requires_grad = False
        for param in self.classif0.parameters():
            param.requires_grad = False
        for param in self.classif1.parameters():
            param.requires_grad = False
        for param in self.classif2.parameters():
            param.requires_grad = False
        for param in self.classif3.parameters():
            param.requires_grad = False


    def freeze_except_SegBranch(self):
        for name, param in self.named_parameters():
            if not name.startswith('SegBranch'):
                param.requires_grad = False
                
                
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'feature_extraction' in name or 'preconv11' in name:
                param.requires_grad = False
                

    # def freeze_except_cost_volume(self):
    #     # 冻结 feature_extraction 和 preconv11
    #     for param in self.feature_extraction.parameters():
    #         param.requires_grad = False
    #     for param in self.preconv11.parameters():
    #         param.requires_grad = False
    #     # 冻结 SegBranch
    #     for param in self.SegBranch.parameters():
    #         param.requires_grad = False
        

    def freeze_except_cost_volume(self):
        # 冻结 feature_extraction 和 preconv11
        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        for param in self.preconv11.parameters():
            param.requires_grad = False
        # 冻结 SegBranch
        for param in self.SegBranch.parameters():
            param.requires_grad = False
        # 冻结 SegBranch 的子模块（如有）
        for param in self.SegBranch.children():
            param.requires_grad = False
        # 检查并打印未被冻结的权重
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'未被冻结的权重: {name}')




    
