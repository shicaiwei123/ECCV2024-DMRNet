import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_multi
from datasets.surf_txt import SURF_generate
from configuration.config_generate_multi import args
import torch
import torch.nn as nn


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
                                 transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False)

    result = calc_accuracy_multi(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/resnet18_dropout_no_seed_no_share_multi.pth"
    args.gpu = 2
    args.modal = 'multi'
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    model = SURF_Baseline(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
