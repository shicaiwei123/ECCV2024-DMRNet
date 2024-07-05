import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
# from models.basic_model import AVClassifier
# from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/home/hudi/data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/home/hudi/data/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')

    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.fusion_method == 'sum':
        out_v = torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)
        out_a = torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)
    else:
        weight_size = model.module.fusion_module.fc_out.weight.size(1)
        out_v = torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)

        out_a = torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)

    print(1)


def get_feature_diversity(a_feature):
    # a_feature=a_feature[0:2,:,:,:]
    # a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
    # a_feature = a_feature.permute(0, 2, 1)  # B HW C
    # a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
    # a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
    # a_std = torch.std(a_feature, dim=2)
    # a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
    # a_similarity = a_similarity / a_std_matrix
    # # print(a_similarity)
    # # a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
    # # print(a_norm.shape)
    # # a_norm = torch.mean(a_norm)
    # a_similarity = a_similarity.view(a_similarity.shape[0] * a_similarity.shape[1] * a_similarity.shape[2], 1)
    # return a_similarity
    G_s_average = 0
    for i in range(a_feature.shape[0]):
        feature_ones = a_feature[i, :]
        # print(feature_ones.shape)
        fm_s = feature_ones.view(feature_ones.shape[0], -1)

        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.t()
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s.t())
        G_s = (G_s / fm_s_normal_factors)

        # print(G_s.shape)

        G_s = G_s.view(G_s.shape[0] * G_s.shape[1], -1)
        G_s = G_s.squeeze(dim=1)
        G_s_average += G_s

    G_s_average = G_s_average / a_feature.shape[0]
    return G_s_average


if __name__ == "__main__":
    # a = torch.rand((64, 512,7,7)).float()
    # b = torch.rand((64, 512,7,7)).float()
    #
    # a_simialr = get_feature_diversity(a)
    # b_simialr = get_feature_diversity(b)
    # print(a_simialr.mean())
    # print(b_simialr.mean())

    # a = torch.tensor([5.6203e+00, 5.3888e+00, 4.2109e+00, 3.0188e-01, 4.2588e+00, 4.4909e-01,
    #                   3.8762e+00, 8.3721e-01, 4.8549e+00, 5.6802e+00, 8.4283e+00, 1.7810e+00,
    #                   8.5022e+00, 1.2992e+00, 4.2480e-01, 3.0763e-04, 1.5757e+00, 1.1833e-01,
    #                   1.1702e+01, 3.4116e+00, 4.1957e+00, 6.5724e+00, 5.9584e+00, 1.1443e+00,
    #                   3.1260e+00, 4.0609e+00, 3.1709e+00, 8.6540e+00, 1.2188e+01, 3.3257e+00,
    #                   5.2703e+00, 5.8784e+00, 4.3645e-02, 1.1004e+01, 6.4454e+00, 1.2809e+00,
    #                   6.3274e+00, 1.2612e+00, 4.2688e-02, 1.9290e+00, 1.0065e+01, 3.0507e+00,
    #                   4.0973e+00, 9.1993e+00, 7.5778e+00, 7.5968e+00, 2.9300e+00, 8.1579e+00,
    #                   8.8068e+00, 8.2280e+00, 4.6077e-01, 8.9611e-01, 1.0298e+01, 1.2361e+00,
    #                   8.6933e+00, 3.0384e+00, 5.5792e+00, 1.3615e+00, 2.7312e-01, 2.6930e+00,
    #                   7.7149e+00, 7.6869e+00, 2.3605e-03, 7.2269e+00])
    # b = torch.tensor([[[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[1.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]],
    #
    #                   [[[0.]]]])
    #
    # c = a * b
    # c = torch.squeeze(c, 1)
    # c = torch.squeeze(c, 1)
    # d = torch.sum(c)
    # e = torch.squeeze(b, 1)
    # e = torch.squeeze(e, 1)
    # e = torch.squeeze(e, 1)
    # f = torch.sum(a * e)
    # ff=a * e
    # g=d/f
    # print(1)



    a=[1,2,3,4,5]
    b=a/3
    print(1)