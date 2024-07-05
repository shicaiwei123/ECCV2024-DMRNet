import torch
import torch.nn as nn
import torchvision.transforms as ttf
import torch.optim as optim
import torch.nn.functional as F
from .backbone import resnet18
import numpy as np
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


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


class ShaSpec_Classfication(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.audio_net = resnet18(modality='visual', args=args)
        self.audio_net.modality = 'audio'
        self.visual_net = resnet18(modality='visual', args=args)

        self.share_encoder = resnet18(modality='visual', args=args)

        self.fusion_module = ConcatFusion(args, output_dim=n_classes)

        self.fusion_projector_1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.fusion_projector_2 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.target_classifier = nn.Linear(1024, n_classes)
        self.unimodal_target_classifier = nn.Linear(512, n_classes)
        self.modality_classifier = nn.Linear(512, 3)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.regurize_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.p = args.p
        self.args = args

    def forward(self, audio, visual):
        audio = audio.repeat(1, 3, 1, 1)
        # print(audio.shape)
        m1_feature_specific = self.audio_net(audio)
        m2_feature_specific = self.visual_net(visual)

        self.share_encoder.modality = 'audio'
        m1_feature_share = self.share_encoder(audio)
        self.share_encoder.modality = 'visual'
        m2_feature_share = self.share_encoder(visual)

        m1_feature_specific = self.regurize_pooling(m1_feature_specific)
        m2_feature_specific = self.regurize_pooling(m2_feature_specific)
        m1_feature_share = self.regurize_pooling(m1_feature_share)
        m2_feature_share = self.regurize_pooling(m2_feature_share)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific
        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share

        m1_feature_specific, m2_feature_specific, p = modality_drop(m1_feature_specific, m2_feature_specific, self.p,
                                                                    self.args)
        data = [m1_feature_share, m2_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m1_feature_share[m1_missing_index] = m2_feature_share[
            m1_missing_index]  # m2和m1的Batch不一样， m2 是visual，可能会多个人frame

        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        # m2_missing_index_for_m1=[x// self.args.use_video_frames for x in m2_missing_index]
        #
        # print(m2_missing_index,m2_missing_index_for_m1)

        m2_feature_share[m2_missing_index] = m1_feature_share[m2_missing_index]

        m1_fusion_out = self.fusion_projector_1(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector_2(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        (_, C, H, W) = m1_fusion_out.size()
        B = m1_fusion_out.size()[0]
        m2_fusion_out = m2_fusion_out.view(B, -1, C, H, W)
        m2_fusion_out = m2_fusion_out.permute(0, 2, 1, 3, 4)

        m1_fusion_out = F.adaptive_avg_pool2d(m1_fusion_out, 1)
        m2_fusion_out = F.adaptive_avg_pool3d(m2_fusion_out, 1)

        m1_fusion_out = torch.flatten(m1_fusion_out, 1)
        m2_fusion_out = torch.flatten(m2_fusion_out, 1)

        fusion_feature = torch.cat([m1_fusion_out, m2_fusion_out], dim=1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat(
            (m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling),
            dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0])], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)
        dco_label = specific_feature_label



        return target_predict, dco_predict, dco_label, m1_feature_share_cache, m2_feature_share_cache, fusion_feature



