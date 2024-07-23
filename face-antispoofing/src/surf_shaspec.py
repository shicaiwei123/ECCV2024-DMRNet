import sys

sys.path.append('..')
from models.shaspec import ShaSpec_Classfication
from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from configuration.config_shaspec import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_shaspec
import torch.optim as optim

import numpy as np
import random


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

    print(type(args.p))

    model = ShaSpec_Classfication(args)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    args.retrain = False
    train_base_multi_shaspec(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)

    for k, v in args.writer_dicts.items():
        v.write('\n')
        v.close()


if __name__ == '__main__':
    deeppix_main(args=args)
