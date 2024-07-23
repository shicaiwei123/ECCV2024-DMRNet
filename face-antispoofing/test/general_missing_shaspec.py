import sys

sys.path.append('..')
from models.shaspec import ShaSpec_Classfication
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_multi
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_former import args
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
        num_workers=4)

    result = calc_accuracy_multi(model=model, loader=test_loader, verbose=True, hter=True)
    print(result[0])
    return result


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    result_model=[]

    for i in range(3):
        i=i
        result_list = []

        pretrain_dir = "../output/models/shaspec_0_dao_0.1_dco_0.02_unimodal_weight_0.0.pth"
        args.gpu = 2
        args.modal = 'multi'
        args.miss_modal = 0
        args.backbone = "resnet18_se"
        print(pretrain_dir)

        for j in range(len(modality_combination)):
            args.p = modality_combination[j]
            print(args.p)
            model = ShaSpec_Classfication(args)
            test_para = torch.load(pretrain_dir)
            model.load_state_dict(torch.load(pretrain_dir))

            result = batch_test(model=model, args=args)
            result_list.append(result)

        result_arr = np.array(result_list,dtype=object)
        result_mean=np.mean(result_arr, axis=0)
        print(result_mean)
        result_model.append(result_mean)
    result_model = np.array(result_model)
