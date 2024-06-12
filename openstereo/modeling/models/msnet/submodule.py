# Copyright 2021 Faranak Shamsafar
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pdb
###############################################################################
""" Fundamental Building Blocks """


###############################################################################


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=pad, bias=False),
        nn.BatchNorm3d(out_channels)
    )


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False)
        )
    else:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class MobileV2_Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileV2_Residual_3D(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio):
        super(MobileV2_Residual_3D, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation_series= [1, 12, 24, 36], padding_series= [1, 6, 12, 18]):
#         super(ASPP, self).__init__()
#         self.conv2d_list = nn.ModuleList()
#         for dilation, padding in zip(dilation_series, padding_series):
#             self.conv2d_list.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
#             )

#         self.image_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.conv2d_1x1_output = nn.Conv2d(len(self.conv2d_list) * out_channels + out_channels, out_channels, kernel_size=1, stride=1)

#     def forward(self, x):
#         size = x.shape[2:] 
#         print("size shape:", size)
#         print("x.shape", x.shape)
#         image_features = self.image_pool(x)
#         print("image_features shape:", image_features)
#         image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
#         print("image_features shape:", image_features)
#         aspp_features = [image_features]
#         print("aspp_features shape:", aspp_features)
#         for conv2d in self.conv2d_list:
#             aspp_features.append(conv2d(x))
#         aspp_features = torch.cat(aspp_features, dim=1)
#         return self.conv2d_1x1_output(aspp_features)

class ASPP(nn.Module):
   def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
      super(ASPP, self).__init__()
      self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
      )
      self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), 
      )
      self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), 
      )
      self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), 
      )
      self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
    #   self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
      self.branch5_relu = nn.ReLU(inplace=True)

      self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),    
      )

   def forward(self, x):
      [b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
      conv1x1 = self.branch1(x)
    #   print("conv1x1.shape:",conv1x1.shape)
      conv3x3_1 = self.branch2(x)
    #   print("conv3x3_1.shape:",conv3x3_1.shape)
      conv3x3_2 = self.branch3(x)
    #   print("conv3x3_2.shape:",conv3x3_2.shape)
      conv3x3_3 = self.branch4(x)
    #   print("conv3x3_3.shape:",conv3x3_3.shape)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
      global_feature = torch.mean(x,2,True)
      global_feature = torch.mean(global_feature,3,True)
    #   print("torch.mean(global_feature,3,True).shape:",global_feature.shape)
      global_feature = self.branch5_conv(global_feature)
    #   print("branch5_conv.shape:",global_feature.shape)
    #   global_feature = self.branch5_bn(global_feature)
    #   print("branch5_bn.shape:",global_feature.shape)
      global_feature = self.branch5_relu(global_feature)
      global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
      #   将五个分支的内容堆叠起来
      #   然后1x1卷积整合特征。
      #-----------------------------------------#
      feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
      result = self.conv_cat(feature_cat)
    #   print("result.shape:",result.shape)
      return result


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes):
        super(SegmentationDecoder, self).__init__()

        # 处理低级特征的卷积层
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 合并特征后的卷积层
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + in_channels, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 分类卷积层
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
    
    
    def forward(self, x, low_level_features):
        # 处理低级特征
        # print("low_level_features shape",low_level_features.size())
        # pdb.set_trace()
        low_level_features = self.shortcut_conv(low_level_features)

        # 合并高级特征和低级特征
        x = torch.cat([x, low_level_features], dim=1)

        # 对合并后的特征进行卷积处理
        x = self.cat_conv(x)

        # 最终分类结果
        x = self.cls_conv(x)    #torch.Size([1, 35, 64, 128])
        
        # x = F.interpolate(x, size=(512, 256), mode='bilinear', align_corners=True)
        # pdb.set_trace()

        return x
###############################################################################
""" Feature Extraction """


###############################################################################
class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样，倍率为2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，核大小和步幅为2
        self.conv = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1)  # 1x1卷积减少通道数

    def forward(self, x):
        x = self.upsample(x)  # 上采样后维度变为 128×128×128
        x = self.maxpool(x)  # 最大池化后维度变为 128×64×128
        x = self.conv(x)  # 通过1x1卷积，减少通道数，维度变为 16×128×256
        return x


class MaxPoolAndStack(nn.Module):
    def __init__(self):
        super(MaxPoolAndStack, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, re1, re2):
        # 对 re2 进行最大池化
        re2_pooled = self.maxpool(re2)
        # 检查最大池化后的 re2 尺寸
        # print(f"re2_pooled shape: {re2_pooled.shape}")
        # 将最大池化后的 re2 和 re1 在通道维度进行堆叠
        output = torch.cat((re1, re2_pooled), dim=1)
        return output


class SegBranch(nn.Module):
    def __init__(self, num_classes):
        super(SegBranch, self).__init__()
        
        self.up_r2 = nn.Sequential(
            nn.Conv2d(128, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.bn_r1 = nn.BatchNorm2d(32)
        
        
        self.final_up_r11 = nn.Sequential(
            nn.ConvTranspose2d(56, 56, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True)
        )
        self.aspp_r3 = ASPP(320, 256)
        
        self.adjust_r4 = nn.Conv2d(32, 48, kernel_size=1)
        

        self.up_r5 = nn.Sequential(
            nn.Conv2d(304, 304, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        self.classifier = nn.Conv2d(360, num_classes, kernel_size=1)
        
    def forward(self, r1, r2, r3, r4):

        r2 = self.up_r2(r2)  # b, 24, w/2, h/2
        
        # r1 processing
        r1 = self.bn_r1(r1)  # b, 32, w/2, h/2
        
        # Stack r1 and r2
        r11 = torch.cat([r1, r2], dim=1)  # b, 56, w/2, h/2
        r11 = self.final_up_r11(r11)  # b, 56, w, h
        
        # r3 processing with ASPP
        r31 = self.aspp_r3(r3)  # b, 256, w/4, h/2
        
        # r4 processing
        r41 = self.adjust_r4(r4)  # b, 48, w/4, h/2
        
        # Stack r31 and r41
        r5 = torch.cat([r31, r41], dim=1)  # b, 304, w/4, h/2
        
        # r5 processing
        r51 = self.up_r5(r5)  # b, 128, w, h
        
        # Stack r11 and r51
        out = torch.cat([r11, r51], dim=1)  # b, 184, w, h
        
        # Final classification
        out = self.classifier(out)  # b, n, w, h
        # pdb.set_trace()
        
        return out


class feature_extraction(nn.Module):
    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 32

        
        
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        
        x = self.firstconv(x)
        # pdb.set_trace()
        ret1 = self.layer1(x) #torch.Size([1, 32, 256, 256])
        l2 = self.layer2(ret1) #torch.Size([1, 64, 128, 128])
        l3 = self.layer3(l2) #torch.Size([1, 128, 64, 128])
        # pdb.set_trace()
        l4 = self.layer4(l3) #torch.Size([1, 256, 32, 32])

        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume, ret1,l3
        # return feature_volume


# class feature_extraction(nn.Module):
#     def __init__(self, add_relus=False):
#         super(feature_extraction, self).__init__()

#         self.expanse_ratio = 3
#         self.inplanes = 32

        
        
#         if add_relus:
#             self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
#                                            nn.ReLU(inplace=True),
#                                            MobileV2_Residual(32, 32, 1, self.expanse_ratio),
#                                            nn.ReLU(inplace=True),
#                                            MobileV2_Residual(32, 32, 1, self.expanse_ratio),
#                                            nn.ReLU(inplace=True)
#                                            )
#         else:
#             self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
#                                            MobileV2_Residual(32, 32, 1, self.expanse_ratio),
#                                            MobileV2_Residual(32, 32, 1, self.expanse_ratio)
#                                            )

#         self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
#         self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
#         self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
#         self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

#     def _make_layer(self, block, planes, blocks, stride, pad, dilation):
#         downsample = None

#         if stride != 1 or self.inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes),
#             )

#         layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

#         return nn.Sequential(*layers)

#     def forward(self, x):

        
#         x = self.firstconv(x)
#         # pdb.set_trace()
#         ret1 = self.layer1(x) #torch.Size([1, 32, 256, 256])
#         l2 = self.layer2(ret1) #torch.Size([1, 64, 128, 128])
#         l3 = self.layer3(l2) #torch.Size([1, 128, 64, 128])
#         # pdb.set_trace()
#         l4 = self.layer4(l3) #torch.Size([1, 256, 32, 32])

#         feature_volume = torch.cat((l2, l3, l4), dim=1)

#         return feature_volume,ret1,l3


###############################################################################
""" Cost Volume Related Functions """


###############################################################################


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:, ::2, :, :] = refimg_fea
    interwoven_features[:, 1::2, :, :] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


###############################################################################
""" Disparity Regression Function """


###############################################################################


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


###############################################################################
""" Loss Function """


###############################################################################


# def model_loss(disp_ests, disp_gt, mask):
#     weights = [0.5, 0.5, 0.7, 1.0]
#     all_losses = []
#     for disp_est, weight in zip(disp_ests, weights):
#         all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
#     return sum(all_losses)
def model_loss(disp_ests, disp_gt, logits, labels, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    combined_loss = CombinedLoss(weight_smooth_l1=weights[0], weight_ce=weights[1])
    loss, _ = combined_loss(disp_ests, disp_gt, logits, labels, mask)
    return loss
