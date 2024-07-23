import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline
from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from configuration.config_baseline_multi import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_baseline
from lib.processing_utils import get_file_list
import torch.optim as optim

import numpy as np
import datetime
import random
import ast


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):
    train_loader = surf_baseline_multi_dataloader(train=True, args=args)
    test_loader = surf_baseline_multi_dataloader(train=False, args=args)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.writer_dicts = {}
    # model_arch_index = open("../output/logs/lib_model_arch_index.txt", 'a+', newline='')
    # args.writer_dicts.update({"model_arch_index": model_arch_index})

    args.epoch = 0
    print(type(args.p))
    try:
        print(args.p)
        p = ast.literal_eval(args.p)
        args.p=p
    except Exception as e:
        print(e)
        print(1)
    # print(args.p)
    # print(type(args.p))
    model = SURF_Baseline(args)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    args.retrain = False
    train_base_multi_baseline(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)

    for k, v in args.writer_dicts.items():
        v.write('\n')
        v.close()


if __name__ == '__main__':
    deeppix_main(args=args)
