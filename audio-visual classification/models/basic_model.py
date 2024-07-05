import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Vanilla,ConcatFusion_Swin
from models.swin_transformer import SwinTransformer
import numpy as np


def modality_drop(x_rgb, x_depth, p, args=None):
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 3, 1 / 3, 1 / 3))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]

    if args.use_video_frames != 1:
        pv = torch.repeat_interleave(p, args.use_video_frames, dim=0)
        # print(pv.shape)
        x_depth = x_depth * pv[:, 1]
    else:
        x_depth = x_depth * p[:, 1]

    return x_rgb, x_depth, p


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(args, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p = [0, 0]
        self.args = args

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        # print(self.p)

        a, v, p = modality_drop(a, v, self.p, args=self.args)

        a_feature = a
        v_feature = v

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        # v = v * 0
        a, v, out, auxi_out, mul, std = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out, auxi_out, mul, std, p


class AVClassifier_Distillation(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Distillation, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion_Vanilla(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p = [0, 0]
        self.args = args

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        # print(self.p)

        a, v, p = modality_drop(a, v, self.p, args=self.args)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a_feature = a
        v_feature = v

        # v = v * 0
        a, v, out, auxi_out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out, auxi_out, a_feature, v_feature, p



class AVClassifier_Distillation_PME(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Distillation_PME, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(args,output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.p = [0, 0]
        self.args = args

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        # print(self.p)

        a, v, p = modality_drop(a, v, self.p, args=self.args)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a_feature = a
        v_feature = v

        # v = v * 0
        x, y, out, auxi_out, mu_dul, std_dul = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out, auxi_out, a_feature, v_feature, mu_dul, std_dul ,p




class AVClassifier_Swin(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Swin, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CEFA':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion_Swin(args, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = SwinTransformer(modality='audio', num_classes=n_classes, in_chans=3)
        self.visual_net = SwinTransformer(modality='visual', num_classes=n_classes)

        print("using pretrain")
        # self.audio_net.load_state_dict(torch.load("swin_tiny_patch4_window7_224_22k.pth")['model'], strict=False)
        # self.visual_net.load_state_dict(torch.load("swin_tiny_patch4_window7_224_22k.pth")['model'], strict=False)
        #
        self.audio_net.load_state_dict(torch.load("swin_base_patch4_window7_224_22k.pth")['model'], strict=False)
        self.visual_net.load_state_dict(torch.load("swin_base_patch4_window7_224_22k.pth")['model'], strict=False)

        self.pe = args.pe
        self.p = [1, 1]
        self.args = args

    def forward(self, audio, visual):

        # print(audio.shape)
        audio = torch.repeat_interleave(audio, 3, 1)
        # print(audio.shape)
        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)
        a=torch.unsqueeze(a,dim=2)
        a=torch.unsqueeze(a,dim=3)
        v=torch.unsqueeze(v,dim=2)
        v=torch.unsqueeze(v,dim=3)

        # print(self.p)

        a, v, p = modality_drop(a, v, self.p, args=self.args)

        a_feature = a
        v_feature = v

        # (_, C, H, W) = v.size()
        # B = a.size()[0]
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)
        #
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool2d(v, 1)
        # v = F.adaptive_avg_pool3d(v, 1)
        #
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        # v = v * 0
        # print(a.shape,v.shape)
        a, v, out, auxi_out, mul, std = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out, auxi_out, mul, std, p
