import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten
from lib.model_arch import modality_drop, unbalance_modality_drop
from lib.PositionalEncoding import LearnedPositionalEncoding, FixedPositionalEncoding
from lib.Transformer import mmTransformerModel


class SURF_Fomer(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.embedding_dim = args.embemdding_dim
        self.seq_length = 14 * 14
        self.dropout_rate = 0.1

        self.linear_project = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        self.restore = []
        self.bns = []
        self.relus = []

        for i in range(3):
            self.bns.append(nn.BatchNorm2d(128))
            self.relus.append(nn.LeakyReLU())
            self.linear_project.append(nn.Conv2d(128, args.embemdding_dim, kernel_size=3, stride=1, padding=1))
            self.restore.append(nn.Conv2d(args.embemdding_dim, 128, kernel_size=3, stride=1, padding=1))
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            self.position_encoding.append(LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            ))

            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(
                mmTransformerModel(modal_num=3, dim=self.embedding_dim, depth=1, heads=8, mlp_dim=4096))

        self.bns = nn.ModuleList(self.bns)
        self.relus = nn.ModuleList(self.relus)
        self.linear_project = nn.ModuleList(self.linear_project)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)
        self.restore = nn.ModuleList(self.restore)

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
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        # print(x_rgb.shape)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        for i in range(3):
            x[i] = self.bns[i](x[i])
            x[i] = self.relus[i](x[i])
            x[i] = self.linear_project[i](x[i])

        for i in range(3):
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])
            x[i] = self._reshape_output(x[i])
            x[i] = self.restore[i](x[i])
        x_rgb = x[0]
        x_ir = x[1]
        x_depth = x[2]

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4

    def _reshape_output(self, x):
        x = x.view(x.size(0), 14, 14, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SURF_Fomer_Auxi(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.embedding_dim = args.embemdding_dim
        self.seq_length = 14 * 14
        self.dropout_rate = 0.1

        self.linear_project = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        self.restore = []
        self.bns = []
        self.relus = []

        for i in range(3):
            self.bns.append(nn.BatchNorm2d(128))
            self.relus.append(nn.LeakyReLU())
            self.linear_project.append(nn.Conv2d(128, args.embemdding_dim, kernel_size=3, stride=1, padding=1))
            self.restore.append(nn.Conv2d(args.embemdding_dim, 128, kernel_size=3, stride=1, padding=1))
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            self.position_encoding.append(LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            ))

            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(
                mmTransformerModel(modal_num=3, dim=self.embedding_dim, depth=1, heads=8, mlp_dim=4096))

        self.bns = nn.ModuleList(self.bns)
        self.relus = nn.ModuleList(self.relus)
        self.linear_project = nn.ModuleList(self.linear_project)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)
        self.restore = nn.ModuleList(self.restore)

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

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        # print(x_rgb.shape)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_ir_out = self.auxi_bone(x_ir)
        x_depth_out = self.auxi_bone(x_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        for i in range(3):
            x[i] = self.bns[i](x[i])
            x[i] = self.relus[i](x[i])
            x[i] = self.linear_project[i](x[i])

        # x_rgb = x[0]
        # x_ir = x[1]
        # x_depth = x[2]

        # if self.drop_mode == 'average':
        #     x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        # else:
        #     x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        # x = [x_rgb, x_ir, x_depth]

        for i in range(3):
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])
            x[i] = self._reshape_output(x[i])
            x[i] = self.restore[i](x[i])
        x_rgb = x[0]
        x_ir = x[1]
        x_depth = x[2]

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        p = torch.sum(p, dim=[2, 3, 4])
        return x,x,x, x_rgb_out, x_ir_out, x_depth_out, p

    def _reshape_output(self, x):
        x = x.view(x.size(0), 14, 14, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SURF_Fomer_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.embedding_dim = args.embemdding_dim
        self.seq_length = 14 * 14
        self.dropout_rate = 0.1

        self.linear_project = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        self.restore = []
        self.bns = []
        self.relus = []

        for i in range(3):
            self.bns.append(nn.BatchNorm2d(128))
            self.relus.append(nn.LeakyReLU())
            self.linear_project.append(nn.Conv2d(128, args.embemdding_dim, kernel_size=3, stride=1, padding=1))
            self.restore.append(nn.Conv2d(args.embemdding_dim, 128, kernel_size=3, stride=1, padding=1))
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            self.position_encoding.append(LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            ))

            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(
                mmTransformerModel(modal_num=3, dim=self.embedding_dim, depth=1, heads=8, mlp_dim=4096))

        self.bns = nn.ModuleList(self.bns)
        self.relus = nn.ModuleList(self.relus)
        self.linear_project = nn.ModuleList(self.linear_project)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)
        self.restore = nn.ModuleList(self.restore)

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

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        # print(x_rgb.shape)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        for i in range(3):
            x[i] = self.bns[i](x[i])
            x[i] = self.relus[i](x[i])
            x[i] = self.linear_project[i](x[i])

        # x_rgb = x[0]
        # x_ir = x[1]
        # x_depth = x[2]

        # if self.drop_mode == 'average':
        #     x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        # else:
        #     x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        # x = [x_rgb, x_ir, x_depth]

        for i in range(3):
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])
            x[i] = self._reshape_output(x[i])
            x[i] = self.restore[i](x[i])
        x_rgb = x[0]
        x_ir = x[1]
        x_depth = x[2]

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(layer3)
        x_rgb_depth = self.auxi_bone(layer3)
        x_depth_out = self.auxi_bone(layer3)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        p = torch.sum(p, dim=[2, 3, 4])
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

    def _reshape_output(self, x):
        x = x.view(x.size(0), 14, 14, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SURF_Fomer_N(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)

        self.seq_length = 14 * 14
        self.dropout_rate = 0.1
        self.args = args

        self.img_dim = 128
        self.embedding_dim = 512
        self.num_channels = 3
        self.dropout_rate = 0.1
        self.attn_dropout_rate = 0.1
        self.input_dim = 128
        positional_encoding_type = 'learned'

        # self.seq_length = self.num_patches              # 序列数量，正常来说是patch的数量，但是，emmm，看后面的实现，patch数量其实就是i HW，所以直接就是HW行了。
        self.seq_length = 14 * 14

        self.shadow_tokens = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        for i in range(self.num_channels):
            self.shadow_tokens.append(torch.zeros(1, self.seq_length, self.embedding_dim).cuda())
            if positional_encoding_type == "learned":
                self.position_encoding.append(LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                ))
            elif positional_encoding_type == "fixed":
                self.position_encoding.append(FixedPositionalEncoding(
                    self.embedding_dim,
                ))
            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(mmTransformerModel(
                modal_num=self.num_channels,
                dim=self.embedding_dim,
                depth=1,
                heads=8,
                mlp_dim=4096,
                dropout_rate=0.1,
                attn_dropout_rate=0.1,
            ))

        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)

        self.inter_position_encoding = LearnedPositionalEncoding(self.seq_length * self.num_channels,
                                                                 self.embedding_dim,
                                                                 self.seq_length * self.num_channels)
        self.inter_pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.fusion = nn.Conv1d(self.seq_length * self.num_channels, self.seq_length, kernel_size=1)

        self.inter_transformer = mmTransformerModel(
            modal_num=3,
            dim=self.embedding_dim,
            depth=1,
            heads=8,
            mlp_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )

        self.conv_x_list = []
        for i in range(self.num_channels):
            self.conv_x_list.append(nn.Conv2d(
                self.input_dim,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ))
        self.conv_x_list = nn.ModuleList(self.conv_x_list)

        self.bn_list = []
        self.relu_list = []
        for i in range(self.num_channels):
            # self.bn_list.append(nn.BatchNorm3d(256))
            self.bn_list.append(nn.BatchNorm2d(128))
            self.relu_list.append(nn.LeakyReLU(inplace=True))
        self.bn_list = nn.ModuleList(self.bn_list)
        self.relu_list = nn.ModuleList(self.relu_list)


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

        self.auxi_bone = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       Flatten(1),
                                       nn.Linear(128, args.class_num))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.class_num)
        self.p=args.p

    def forward(self, img_rgb, img_ir, img_depth):

        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_ir_out = self.auxi_bone(x_ir)
        x_depth_out = self.auxi_bone(x_depth)

        # print(x_rgb.shape)

        x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        # 拆分patch
        for i in range(self.num_channels):
            x[i] = self.bn_list[i](x[i])
            x[i] = self.relu_list[i](x[i])
            x[i] = self.conv_x_list[i](x[i])
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)

        # intra-modality
        for i in range(self.num_channels):
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])

        x_rgb = x[0]
        x_ir = x[1]
        x_depth = x[2]    # B L C  64  196  512

        # print(x_rgb.shape,x_ir.shape,x_depth.shape)

        # inter-modality

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        x = self.inter_position_encoding(x)
        x = self.inter_pe_dropout(x)
        x = self.inter_transformer(x)

        x = self.fusion(x)

        x = self._reshape_output(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x,x,x, x_rgb_out, x_ir_out, x_depth_out,p

    def _reshape_output(self, x):
        x = x.view(x.size(0), 14, 14, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
