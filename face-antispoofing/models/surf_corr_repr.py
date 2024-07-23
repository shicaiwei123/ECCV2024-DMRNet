import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten
from lib.model_arch import modality_drop, unbalance_modality_drop


class Corr_Repr(nn.Module):
    def __init__(self):
        super(Corr_Repr, self).__init__()
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.mpe1 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 3))
        self.mpe2 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 3))
        self.mpe3 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 3))

    def forward(self, x_rgb, x_ir, x_depth):
        x_rgb_t = self.average_pooling(x_rgb)
        x_ir_t = self.average_pooling(x_ir)
        x_depth_t = self.average_pooling(x_depth)

        x_rgb_t = x_rgb_t.view(x_rgb_t.shape[0], -1)
        x_ir_t = x_ir_t.view(x_ir_t.shape[0], -1)
        x_depth_t = x_depth_t.view(x_depth_t.shape[0], -1)

        # print(x_rgb.shape)

        x_rgb_func = self.mpe1(x_rgb_t)
        x_ir_func = self.mpe2(x_ir_t)
        x_depth_func = self.mpe3(x_depth_t)

        x_rgb_func = torch.unsqueeze(x_rgb_func, 2)
        x_rgb_func = torch.unsqueeze(x_rgb_func, 3)
        x_rgb_func = torch.unsqueeze(x_rgb_func, 4)

        x_ir_func = torch.unsqueeze(x_ir_func, 2)
        x_ir_func = torch.unsqueeze(x_ir_func, 3)
        x_ir_func = torch.unsqueeze(x_ir_func, 4)

        x_depth_func = torch.unsqueeze(x_depth_func, 2)
        x_depth_func = torch.unsqueeze(x_depth_func, 3)
        x_depth_func = torch.unsqueeze(x_depth_func, 4)

        # print(x_rgb_func[:, 0].shape)

        x_rgb_fuse = x_ir * x_rgb_func[:, 0] + x_depth * x_rgb_func[:, 1] + x_rgb_func[:, 2]
        x_ir_fuse = x_rgb * x_ir_func[:, 0] + x_depth * x_ir_func[:, 1] + x_ir_func[:, 2]
        x_depth_fuse = x_rgb * x_depth_func[:, 0] + x_ir * x_depth_func[:, 1] + x_depth_func[:, 2]

        return x_rgb_fuse, x_ir_fuse, x_depth_fuse


class SURF_CR(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.corr_repr = Corr_Repr()
        self.args=args


        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_ir, img_depth):
        # x_rgb = self.special_bone_rgb(img_rgb)
        # x_ir = self.special_bone_ir(img_ir)
        # x_depth = self.special_bone_depth(img_depth)
        #
        # if self.drop_mode == 'average':
        #     x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p)
        # else:
        #     x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p)

        if self.drop_mode == 'average':
            img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        else:
            img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)

        if torch.sum(img_rgb) == 0:
            x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        else:
            x_rgb = self.special_bone_rgb(img_rgb)

        if torch.sum(img_ir) == 0:
            x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        else:
            x_ir = self.special_bone_ir(img_ir)

        if torch.sum(img_depth) == 0:
            x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        else:
            x_depth = self.special_bone_depth(img_depth)

        x_rgb, x_ir, x_depth = self.corr_repr(x_rgb, x_ir, x_depth)
        # print(x_rgb.shape)

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4

class SURF_CMFL(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.special_bone_rgb = resnet18_se(args, pretrained=False)
        self.special_bone_depth = resnet18_se(args, pretrained=False)
        self.special_bone_ir = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode

        self.args=args



    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        return x_rgb, x_ir, x_depth
