import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
from dataset.KSDataset import KSDataset

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier,AVClassifier_Distillation,AVClassifier_Swin
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F
from models.shaspec import ShaSpec_Classfication

import csv


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=1, type=int)
    parser.add_argument('--audio_path', default='./data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha',default=0.0, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', default='', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--pme', type=int, default=1)

    return parser.parse_args()


def get_feature_diff(x1, x2):
    # print(x1.shape,x2.shape)
    x1 = F.adaptive_avg_pool2d(x1, (7, 7))
    x2 = F.adaptive_avg_pool2d(x2, (7, 7))
    # x1 = torch.mean(x1, dim=(2, 3))
    # x2 = torch.mean(x2, dim=(2, 3))

    x1 = x1.permute(0, 2, 3, 1).contiguous()
    x2 = x2.permute(0, 2, 3, 1).contiguous()

    x1 = x1.view(-1, x1.shape[3])
    x2 = x2.view(-1, x2.shape[3])

    print("mse:", F.mse_loss(x1, x2))
    mse_diff = F.mse_loss(x1, x2, reduction='none')
    # mse_diff = (mse_diff.max() - mse_diff) / (mse_diff.max() - mse_diff.min())
    mse_diff = mse_diff.view(-1, 1)
    mse_diff = mse_diff.squeeze(dim=1)
    # print(mse_diff.shape)

    # diff = F.mse_loss(rgb, depth)
    # diff = torch.cosine_similarity(rgb, depth)
    # diff = torch.mean(diff)
    # print(simi.shape)

    diff = get_feature_diversity_two(x1, x2)

    return diff, mse_diff


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
        G_s = (G_s / (fm_s_normal_factors + 1e-16))

        # print(G_s.shape)

        G_s = G_s.view(G_s.shape[0] * G_s.shape[1], -1)
        G_s = G_s.squeeze(dim=1)
        G_s_average += G_s

    G_s_average = G_s_average / a_feature.shape[0]
    return G_s_average


def get_feature_diversity_two(a_feature, v_feature):
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
        # if i>2:
        #     break
        a_feature_ones = a_feature[i, :]
        v_feature_ones = v_feature[i, :]
        # print(feature_ones.shape)
        fm_s = a_feature_ones.view(a_feature_ones.shape[0], -1)
        fm_s_trans = v_feature_ones.view(v_feature_ones.shape[0], -1)
        fm_s_trans = fm_s_trans.t()

        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))

        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s_trans)
        G_s = (G_s / (fm_s_normal_factors + 1e-16))

        # print(G_s.shape)

        G_s = G_s.view(G_s.shape[0] * G_s.shape[1], -1)
        G_s = G_s.squeeze(dim=1)
        G_s_average += G_s

    G_s_average = G_s_average / a_feature.shape[0]
    return G_s_average


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        a_save = torch.zeros(512 * 512).cuda().float()
        v_save = torch.zeros(512 * 512).cuda().float()
        c = 0
        d = 0
        mse_diff_save = torch.zeros(512 * 64 * 49).cuda().float()
        for step, (spec, image, label) in enumerate(dataloader):
            print(step)
            # if step < 5:
            #     continue
            # if step > :
            #     break

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out, auxi_out, mul, std, p = model(spec.unsqueeze(1).float(), image.float())

            prediction = softmax(out)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())

                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0

        return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def single_modal_main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    args.p = [0, 1]

    model = AVClassifier(args)
    # model=AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model_path = "/home/data2/shicaiwei/results/best_model_of_dataset_CREMAD_Normal_alpha_0.8_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_79_acc_0.5779569892473119.pth"

    model.load_state_dict(torch.load(model_path)['model'], strict=False)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

    print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


def two_model_valid(args, audio_model, visual_model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        audio_model.eval()
        visual_model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        a_save = torch.zeros(512 * 512).cuda().float()
        v_save = torch.zeros(512 * 512).cuda().float()
        c = 0
        d = 0

        mse_diff_save = torch.zeros(64 * 49 * 512).cuda().float()

        for step, (spec, image, label) in enumerate(dataloader):
            # print(step)
            if step > 10:
                break

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out, a_feature, _, _, _, _, _ = audio_model(spec.unsqueeze(1).float(), image.float())
            a, v, out, _, v_feature, _, _, _, _ = visual_model(spec.unsqueeze(1).float(), image.float())

            feature_similar, mse_diff = get_feature_diff(a_feature, v_feature)
            mse_diff_save += mse_diff
            print(feature_similar.mean())
            c += feature_similar.mean()
            d += mse_diff.mean()

        print(c, d)
        mse_diff_save = mse_diff_save / (11)
        # file_name = 'audio_visual_similar_in_unimodal' + '.csv'
        # with open(file_name, 'a', newline='') as f:
        #     for i in range(feature_similar.shape[0]):
        #         writer = csv.writer(f)
        #         row = [feature_similar[i].cpu().detach().numpy()]
        #         writer.writerow(row)

        # a_similarity = a_save / 6
        # v_similarity = v_save / 6

        # file_name = 'audio_in_multimodal_baseline' + '.csv'
        # with open(file_name, 'a', newline='') as f:
        #     for i in range(a_similarity.shape[0]):
        #         writer = csv.writer(f)
        #         row = [a_similarity[i].cpu().detach().numpy()]
        #         writer.writerow(row)
        #
        #
        # file_name = 'visual_in_multimodal_baseline' + '.csv'
        # with open(file_name, 'a', newline='') as f:
        #     for i in range(v_similarity.shape[0]):
        #         writer = csv.writer(f)
        #         row = [a_similarity[i].cpu().detach().numpy()]
        #         writer.writerow(row)

        # file_name = 'visual_in_unimodal' + '.csv'
        # with open(file_name, 'a', newline='') as f:
        #     for i in range(v_similarity.shape[0]):
        #         writer = csv.writer(f)
        #         row = [a_similarity[i].cpu().detach().numpy()]
        #         writer.writerow(row)

        file_name = 'mse_diff_unimodal' + '.csv'
        with open(file_name, 'a', newline='') as f:
            for i in range(mse_diff_save.shape[0]):
                writer = csv.writer(f)
                row = [mse_diff_save[i].cpu().detach().numpy()]
                writer.writerow(row)

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)




def general_model_test():
    softmax = nn.Softmax(dim=1)

    args = get_arguments()
    print(args)

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

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)
    model.p = [1, 1]

    # model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # model_path = "./results/ks/pme_share/best_model_of_dataset_KineticSound_Normal_beta_0.5_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_76_acc_0.5744029393753828.pth"
    # model_path = "./results/cramed/baseline/best_model_of_dataset_CREMAD_Normal_alpha_0.0_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_88_acc_0.5793010752688172.pth"
    # model_path='./results/best_model_of_dataset_KineticSound_Normal_alpha_0.0_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_81_acc_0.5266380894060012.pth'
    # model_path = './results/cramed/pme/best_model_of_dataset_CREMAD_Normal_alpha_0.001_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_88_acc_0.6344086021505376.pth'
    # model_path = './results/cramed/pme/best_model_of_dataset_CREMAD_Normal_alpha_0.001_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_98_acc_0.6209677419354839.pth'
    model_path="./results/cramed/pme/best_model_of_dataset_CREMAD_Normal_beta_0.07_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_98_acc_0.6384408602150538.pth"

    model.load_state_dict(torch.load(model_path)['model'], strict=True)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    modality_combinations = [[1, 0], [0, 1], [1, 1]]
    for p in modality_combinations:
        model.module.p = p

        with torch.no_grad():
            model.eval()
            print(model.module.p)
            # TODO: more flexible
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]

            for step, (spec, image, label) in enumerate(test_dataloader):
                print(step)
                # if step < 5:
                #     continue
                # if step > :
                #     break

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                a, v, out, auxi_out, mul, std, p = model(spec.unsqueeze(1).float(), image.float())

                prediction = softmax(out)

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())

                    num[label[i]] += 1.0

                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0

        print(sum(acc) / sum(num))


def general_model_test_shaspec():
    softmax = nn.Softmax(dim=1)

    args = get_arguments()
    print(args)

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

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    args.p = [1, 1]

    # model = AVClassifier(args)
    # model.p=[1,0]
    # model=AVClassifier(args)
    model = ShaSpec_Classfication(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model_path = "./results/cramed/shaspec/best_model_of_dataset_CREMAD_Normal_dco_0.02_dao_0.1_unimodal_2.0_epoch_80_acc_0.5591397849462365.pth"
    # model_path="./results/cramed/shaspec/best_model_of_dataset_KineticSound_Normal_dco_0.02_dao_0.5_unimodal_0.0_epoch_89_acc_0.5407225964482547.pth"
    model.load_state_dict(torch.load(model_path)['model'], strict=False)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    modality_combinations = [[1, 0], [0, 1], [1, 1]]
    for p in modality_combinations:
        model.module.p = p

        with torch.no_grad():
            model.eval()
            print(model.module.p)
            # TODO: more flexible
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]

            for step, (spec, image, label) in enumerate(test_dataloader):
                print(step)
                # if step < 5:
                #     continue
                # if step > :
                #     break

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                out, dco_predict, dco_label, m1_feature_share_cache, m2_feature_share_cache, fusion_feature, m1_predict, m2_predict = model(
                    spec.unsqueeze(1).float(), image.float())

                prediction = softmax(out)

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())

                    num[label[i]] += 1.0

                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0

        print(sum(acc) / sum(num))



def general_model_test_distillation():
    softmax = nn.Softmax(dim=1)

    args = get_arguments()
    print(args)

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

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    args.p = [1, 1]

    # model = AVClassifier(args)
    # model.p=[1,0]
    # model=AVClassifier(args)
    model = AVClassifier_Distillation(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model_path = "./results/ks/mmanet/best_model_of_dataset_KineticSound_Normal_alpha_0.001_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_73_acc_0.5358236374770361.pth"
    # model_path = "./results/cramed/mmanet/best_model_of_dataset_CREMAD_Normal_alpha_0.001_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_88_acc_0.5981182795698925.pth"
    # model_path = "./results/cramed/mmanet_kl_1.0/best_model_of_dataset_CREMAD_Normal_alpha_0.001_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_88_acc_0.5994623655913979.pth"
    # model_path="./results/cramed/shaspec/best_model_of_dataset_KineticSound_Normal_dco_0.02_dao_0.5_unimodal_0.0_epoch_89_acc_0.5407225964482547.pth"
    model.load_state_dict(torch.load(model_path)['model'], strict=False)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    modality_combinations = [[1, 0], [0, 1], [1, 1]]
    for p in modality_combinations:
        model.module.p = p

        with torch.no_grad():
            model.eval()
            print(model.module.p)
            # TODO: more flexible
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]

            for step, (spec, image, label) in enumerate(test_dataloader):
                print(step)
                # if step < 5:
                #     continue
                # if step > :
                #     break

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                a, v, out, auxi_out, mul, std, p = model(
                    spec.unsqueeze(1).float(), image.float())

                prediction = softmax(out)

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())

                    num[label[i]] += 1.0

                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0

        print(sum(acc) / sum(num))


if __name__ == '__main__':
    # two_modal_main()
    # single_modal_main()
    general_model_test()
    # general_model_test_shaspec()
    # general_model_test_distillation()