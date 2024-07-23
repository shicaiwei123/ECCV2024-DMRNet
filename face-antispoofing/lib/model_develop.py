'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
import time

import os
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy


def calc_accuracy_multi_advisor(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for sample_batch in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_rgb, ir_target, depth_target, binary_target = sample_batch['image_rgb'], sample_batch['image_ir'], \
            sample_batch['image_depth'], sample_batch['binary_label']

        if torch.cuda.is_available():
            img_rgb, ir_target, depth_target, binary_target = img_rgb.cuda(), ir_target.cuda(), depth_target.cuda(), binary_target.cuda()

        with torch.no_grad():
            if args.method == 'deeppix':
                ir_out, depth_out, outputs_batch = model(img_rgb)
            elif args.method == 'pyramid':
                if args.origin_deeppix:
                    x, x, x, x, outputs_batch = model(img_rgb)
                else:
                    x, x, x, x, outputs_batch = model(img_rgb)
            else:
                print("test error")
        outputs_full.append(outputs_batch)
        labels_full.append(binary_target)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        FRR = living_wrong / (living_wrong + living_right)
        APCER = living_wrong / (spoofing_right + living_wrong)
        NPCER = spoofing_wrong / (spoofing_wrong + living_right)
        FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
        HTER = (FAR + FRR) / 2

        FAR = float("%.6f" % FAR)
        FRR = float("%.6f" % FRR)
        HTER = float("%.6f" % HTER)
        accuracy = float("%.6f" % accuracy)

        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_multi(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    mul_full = []
    std_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batchs = model(img_rgb, img_ir, img_depth)
            if isinstance(outputs_batchs, tuple):
                outputs_batch = outputs_batchs[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)



    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)

    # if std_concat:
    #     std_full = torch.cat(std_full, dim=0)
    #     # mul_full = torch.cat(mul_full, dim=0)

    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            APCER = living_wrong / (living_wrong + living_right)
            NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)

            ACER = (APCER + NPCER) / 2

            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong), (
                outputs_full, labels_full, mul_full, std_full)
            return [accuracy, 0, 0, 0, 0, 0], (outputs_full, labels_full, mul_full, std_full)

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, APCER, NPCER, ACER], (outputs_full, labels_full, mul_full, std_full)
    else:
        return [accuracy], (outputs_full, labels_full, mul_full, std_full)


def train_base_multi_baseline(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]

    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            if isinstance(output, tuple):
                output = output[0]
            loss_cls = cost(output, target)

            loss = loss_cls

            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]

    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            (mul, std) = output[-1]
            p = output[1]

            if isinstance(output, tuple):
                output = output[0]
            loss_cls = cost(output, target)
            if torch.sum(std) == 0:  # 正常训练。没有分布化
                loss = loss_cls
                loss_kl_sum += 0
            else:

                variance_dul = std ** 2
                variance_dul = variance_dul.view(variance_dul.shape[0], -1)
                mul = mul.view(mul.shape[0], -1)
                loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
                loss_kl = torch.mean(loss_kl)

                p = p.cpu().detach()
                p = p.view(p.shape[0], -1)
                variance_dul = torch.mean(variance_dul, 1)
                for i in range(len(modality_combination)):
                    # print(p.shape)
                    index = (p == modality_combination[i])
                    index = index[:, 0] & index[:, 1] & index[:, 2]
                    # print(index)
                    # print(index.shape)
                    varaince_slect = variance_dul[index]
                    varaince_list[i] += (torch.mean(varaince_slect)).cpu().detach().numpy()

                if epoch > 5:
                    loss = loss_cls + args.kl_scale * loss_kl
                else:
                    loss = loss_cls

                loss_kl_sum += loss_kl.item()

            train_loss += loss.item()

            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_shaspec(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]
    cls_sum = 0
    dco_loss_sum = 0
    dao_loss_sum = 0
    unimodal_loss_sum = 0
    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            target_predict, dco_predict, dco_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache, fusion_feature= model(
                img_rgb, img_ir, img_depth)

            task_loss = cost(target_predict, target)



            dao_loss = mse_func(m1_feature_share_cache, m2_feature_share_cache) + mse_func(m2_feature_share_cache,
                                                                                           m3_feature_share_cache)

            dco_loss = cost(dco_predict, dco_label)

            loss = task_loss + args.dao_weight * dao_loss + args.dco_weight * dco_loss

            train_loss += loss.item()

            loss.backward()

            cls_sum += task_loss.item()
            dao_loss_sum += dao_loss.item()
            dco_loss_sum += dco_loss.item()

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(cls_sum / len(train_loader), dao_loss_sum / len(train_loader), dco_loss_sum / len(train_loader),
              unimodal_loss_sum / len(train_loader))
        cls_sum = 0
        dco_loss_sum = 0
        dao_loss_sum = 0
        unimodal_loss_sum = 0

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)




from random import shuffle


def train_base_multi_mix(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 1:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            (mul, std) = output[-1]

            if isinstance(output, tuple):
                output = output[0]

            mul = mul.view(mul.shape[0], -1)
            mul = torch.mean(mul, dim=1)
            std = std.view(std.shape[0], -1)
            std = torch.mean(std, dim=1)

            c = list(zip(output[:, 0], output[:, 1], target, mul, std))
            shuffle(c)
            shuffle_output1, shuffle_output_2, shuffle_target, shuffle_mul, shuffle_std = zip(*c)
            shuffle_output1 = torch.tensor(shuffle_output1).cuda().unsqueeze(dim=1)
            shuffle_output_2 = torch.tensor(shuffle_output_2).cuda().unsqueeze(dim=1)
            # print(shuffle_output1)

            shuffle_output = torch.cat((shuffle_output1, shuffle_output_2), dim=1)

            # print(shuffle_output)
            shuffle_target = torch.tensor(shuffle_target).cuda()
            shuffle_mul = torch.tensor(shuffle_mul).cuda()
            shuffle_std = torch.tensor(shuffle_std).cuda()
            # shuffle_output=[torch.tensor(data) for data in shuffle_output]

            std1 = std / (shuffle_std + std)
            std2 = shuffle_std / (std + shuffle_std)
            std1 = std1.unsqueeze(dim=1)
            std2 = std2.unsqueeze(dim=1)
            # print(std1.shape)
            # print(std2.shape)
            # print(std1)
            # print(std2)

            output_mix = std1 * output + std2 * shuffle_output

            loss_cls_1 = cost(output_mix, target)

            loss_cls_2 = cost(output_mix, shuffle_target)

            loss_cls = loss_cls_1 + loss_cls_2

            if torch.sum(std) == 0:  # 正常训练。没有分布化
                loss = loss_cls
                loss_kl_sum += 0
            else:

                variance_dul = std ** 2
                variance_dul = variance_dul.view(variance_dul.shape[0], -1)
                mul = mul.view(mul.shape[0], -1)
                loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
                loss_kl = torch.mean(loss_kl)
                loss = loss_cls + args.kl_scale * loss_kl
                loss_kl_sum += loss_kl.item()

            train_loss += loss.item()

            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def get_std(std, index):
    index1 = index.view(index.shape[0], -1)
    index1 = torch.squeeze(index1, dim=1)
    std = std ** 2
    std = std.view(std.shape[0], -1)
    std = torch.mean(std, dim=1)
    # print(std.shape,index1.shape)
    std1 = torch.sum(std * index1) / torch.sum(index1)
    return std1


def train_base_multi_auxi_dul(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    loss_kl_sum = 0
    log_list = []  # log need to save
    auxi_sum = 0
    fusion_sum = 0

    std1_sum = 0
    std2_sum = 0
    std3_sum = 0
    std4_sum = 0
    std5_sum = 0
    std6_sum = 0
    std7_sum = 0

    loss1_sum = 0
    loss2_sum = 0
    loss3_sum = 0
    loss4_sum = 0
    loss5_sum = 0
    loss6_sum = 0
    loss7_sum = 0

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p, (mul, std) = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            std1 = get_std(std, (p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))
            std2 = get_std(std, (1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
            std3 = get_std(std, (1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
            std4 = get_std(std, (p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
            std5 = get_std(std, (p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
            std6 = get_std(std, (1 - p[:, 0]) * (p[:, 1]) * (p[:, 2]))
            std7 = get_std(std, (p[:, 0]) * (p[:, 1]) * (p[:, 2]))

            std1_sum += std1.item()
            std2_sum += std2.item()
            std3_sum += std3.item()
            std4_sum += std4.item()
            std5_sum += std5.item()
            std6_sum += std6.item()
            std7_sum += std7.item()

            count1 += torch.sum((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2])).cpu().numpy()
            count2 += torch.sum((1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2])).cpu().numpy()
            count3 += torch.sum((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2])).cpu().numpy()
            count4 += torch.sum((p[:, 0]) * (p[:, 1]) * (1 - p[:, 2])).cpu().numpy()
            count5 += torch.sum((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2])).cpu().numpy()
            count6 += torch.sum((1 - p[:, 0]) * (p[:, 1]) * (p[:, 2])).cpu().numpy()
            count7 += torch.sum((p[:, 0]) * (p[:, 1]) * (p[:, 2])).cpu().numpy()

            if args.dataset == 'surf':

                fusion_loss = auxi_cross_entropy(output, target)

                loss1 = get_std(fusion_loss, (p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))
                loss2 = get_std(fusion_loss, (1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
                loss3 = get_std(fusion_loss, (1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
                loss4 = get_std(fusion_loss, (p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
                loss5 = get_std(fusion_loss, (p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
                loss6 = get_std(fusion_loss, (1 - p[:, 0]) * (p[:, 1]) * (p[:, 2]))
                loss7 = get_std(fusion_loss, (p[:, 0]) * (p[:, 1]) * (p[:, 2]))

                loss1_sum += loss1.item()
                loss2_sum += loss2.item()
                loss3_sum += loss3.item()
                loss4_sum += loss4.item()
                loss5_sum += loss5.item()
                loss6_sum += loss6.item()
                loss7_sum += loss7.item()

                fusion_loss = fusion_loss.mean()

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()

                x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            else:

                fusion_loss = cost(output, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]))) / p.shape[0]
                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                x_ir_loss = torch.sum(x_ir_loss_batch * ((1 - p[:, 2]) * (p[:, 1]) * (1 - p[:, 0]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 1]) * (p[:, 2]) * (1 - p[:, 0]))) / p.shape[0]

            if epoch > 5:

                loss_cls = fusion_loss + args.auxi_scale * (x_rgb_loss + x_depth_loss + x_ir_loss)
            else:
                loss_cls = fusion_loss

            fusion_sum += fusion_loss.cpu().detach().numpy()
            auxi_sum += (x_rgb_loss + x_depth_loss + x_ir_loss).cpu().detach().numpy()

            variance_dul = std ** 2
            variance_dul = variance_dul.view(variance_dul.shape[0], -1)
            mul = mul.view(mul.shape[0], -1)

            loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
            loss_kl = torch.mean(loss_kl)



            loss = loss_cls + args.kl_scale * loss_kl

            train_loss += loss.item()
            loss_kl_sum += loss_kl.item()
            loss.backward()

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        acer_test = result_test[-1]
        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 30:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))

        print(fusion_sum / len(train_loader), loss_kl_sum / len(train_loader), auxi_sum / (len(train_loader)))

        print('std:', std1_sum / count1, std2_sum / count2, std3_sum / count3,
              std4_sum / count4, std5_sum / count5, std6_sum / count6,
              std7_sum / count7)

        print('loss:', loss1_sum / count1, loss2_sum / count2, loss3_sum / count3,
              loss4_sum / count4, loss5_sum / count5, loss6_sum / count6,
              loss7_sum / count7)

        print('count:', count1, count2, count3, count4, count5, count6, count7)
        loss_kl_sum = 0
        train_loss = 0
        auxi_sum = 0
        fusion_sum = 0
        std1_sum = 0
        std2_sum = 0
        std3_sum = 0
        std4_sum = 0
        std5_sum = 0
        std6_sum = 0
        std7_sum = 0

        loss1_sum = 0
        loss2_sum = 0
        loss3_sum = 0
        loss4_sum = 0
        loss5_sum = 0
        loss6_sum = 0
        loss7_sum = 0

        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)

