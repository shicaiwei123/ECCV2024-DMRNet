import torch
import torch.nn as nn
from models.resnet18_se import resnet18_se

from lib.model_arch import modality_drop, unbalance_modality_drop


class ShaSpec_Classfication(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

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

        self.fusion_module = nn.Sequential(model_resnet18_se_1.layer3_new,
                                           model_resnet18_se_1.layer4,
                                           model_resnet18_se_1.avgpool,
                                           )

        self.share_encoder = nn.Sequential(model_resnet18_se_4.conv1,
                                           model_resnet18_se_4.bn1,
                                           model_resnet18_se_4.relu,
                                           model_resnet18_se_4.maxpool,
                                           model_resnet18_se_4.layer1,
                                           model_resnet18_se_4.layer2,
                                           model_resnet18_se_4.se_layer)

        self.fusion_projector_1 = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion_projector_2 = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion_projector_3 = nn.Conv2d(256, 128, 1, 1, 0)
        self.target_classifier = nn.Linear(512, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 3)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, img_rgb, img_ir, img_depth):
        m1_feature_specific = self.special_bone_rgb(img_rgb)
        m2_feature_specific = self.special_bone_ir(img_ir)
        m3_feature_specific = self.special_bone_depth(img_depth)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific
        m3_feature_specific_cache = m3_feature_specific

        m1_feature_share = self.share_encoder(img_rgb)
        m2_feature_share = self.share_encoder(img_ir)
        m3_feature_share = self.share_encoder(img_depth)

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share
        m3_feature_share_cache = m3_feature_share

        m1_feature_specific, m2_feature_specific, m3_feature_specific, p = modality_drop(m1_feature_specific,
                                                                                         m2_feature_specific,
                                                                                         m3_feature_specific, self.p,
                                                                                         args=self.args)
        data = [m1_feature_share, m2_feature_share, m3_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share, m3_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m3_missing_index = [not bool(i) for i in (torch.sum(m3_feature_share, dim=[1, 2, 3]))]

        m1_feature_share[m1_missing_index] = (m2_feature_share[m1_missing_index] + m3_feature_share[
            m1_missing_index])
        m1_c = (((~(torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m1_c = m1_c.unsqueeze(1)
        m1_c = m1_c.unsqueeze(2)
        m1_c = m1_c.unsqueeze(3)
        m1_c = m1_c.repeat(1, 128, 14, 14).cuda()
        m1_feature_share[m1_missing_index] = m1_feature_share[m1_missing_index] / m1_c[m1_missing_index]

        m2_feature_share[m2_missing_index] = (m1_feature_share[m2_missing_index] + m3_feature_share[
            m2_missing_index])
        m2_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m2_c = m2_c.unsqueeze(1)
        m2_c = m2_c.unsqueeze(2)
        m2_c = m2_c.unsqueeze(3)
        m2_c = m2_c.repeat(1, 128, 14, 14).cuda()
        m2_feature_share[m2_missing_index] = m2_feature_share[m2_missing_index] / m2_c[m2_missing_index]

        m3_feature_share[m3_missing_index] = (m1_feature_share[m3_missing_index] + m3_feature_share[
            m3_missing_index])
        m3_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m2_missing_index))).float() + 1))
        m3_c = m3_c.unsqueeze(1)
        m3_c = m3_c.unsqueeze(2)
        m3_c = m3_c.unsqueeze(3)
        m3_c = m3_c.repeat(1, 128, 14, 14).cuda()
        m3_feature_share[m3_missing_index] = m3_feature_share[m3_missing_index] / m3_c[m3_missing_index]

        m1_fusion_out = self.fusion_projector_1(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector_2(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        m3_fusion_out = self.fusion_projector_3(torch.cat((m3_feature_specific, m3_feature_share), dim=1))
        m3_fusion_out = m3_fusion_out + m3_feature_specific

        fusion_feature = self.fusion_module(torch.cat([m1_fusion_out, m2_fusion_out, m3_fusion_out], dim=1))

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)
        m3_feature_specific_cache_pooling = self.pooling(m3_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        m3_feature_specific_cache_pooling = m3_feature_specific_cache_pooling.view(
            m3_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat(
            (m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling, m3_feature_specific_cache_pooling),
            dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]) + 1], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)
        dco_label = specific_feature_label



        return target_predict, dco_predict, dco_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache, fusion_feature
