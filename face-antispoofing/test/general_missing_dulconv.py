import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi,SURF_Baseline_ConvDUL_Auxi_Share
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_multi
from datasets.surf_txt import SURF
from configuration.config_baseline_multi import args
import torch
import torch.nn as nn
import os
import numpy as np


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "../data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8)

    result, para = calc_accuracy_multi(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)
    return result



def performance_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    result_model = []

    for i in range(3):
        i = i
        result_list = []

        pretrain_dir = "../output/models/multi_dulconv_kl_auxi_share_0.7_" + str(
            i) + "_kl_scale_0.001_auxi_scale_0.7_average_acer_best_.pth"
        args.gpu = 2
        args.modal = 'multi'
        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.inplace_new = 384
        print(pretrain_dir)

        for j in range(len(modality_combination)):
            args.p = modality_combination[j]
            print(args.p)
            model = SURF_Baseline_ConvDUL_Auxi_Share(args)
            # model=SURF_Baseline_DULConv(args)
            test_para = torch.load(pretrain_dir)
            model.load_state_dict(torch.load(pretrain_dir))

            result = batch_test(model=model, args=args)
            result_list.append(result)

        result_arr = np.array(result_list)
        result_mean = np.mean(result_arr, axis=0)
        print(result_mean)
        result_model.append(result_mean)
    result_model = np.array(result_model)
    print(np.mean(result_model, axis=0))


if __name__ == '__main__':
    performance_test()
