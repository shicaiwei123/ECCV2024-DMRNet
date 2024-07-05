from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as ttf
import numpy as np
import cv2

from dataset.cefa_dataset_class import *


# def sort_by_index():

def sort_via_assist(origin_list, assist_list):
    '''
    利用assist_list 帮助排序 origin_list
    :param sort_list:
    :param assist_list:
    :return:
    '''

    zipped = zip(origin_list, assist_list)
    sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]))
    result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in result]

    return x_axis, y_axis


class CEFA_Multi(Dataset):
    def __init__(self, args, mode, protocol, miss_modal, fill=0, transform=None):
        self.args = args
        cefa_dataset_rgb = load_casia_race(args.data_root, modal='profile', protocol=protocol,
                                           mode=mode)
        cefa_dataset_depth = load_casia_race(args.data_root, modal='depth', protocol=protocol,
                                             mode=mode)
        cefa_dataset_ir = load_casia_race(args.data_root, modal='ir', protocol=protocol,
                                          mode=mode)

        print(len(cefa_dataset_ir), len(cefa_dataset_depth), len(cefa_dataset_rgb))

        self.image_list_rgb, self.label_list_rgb = get_sframe_paths_labels(cefa_dataset_rgb, phase=mode, ratio=1)
        self.image_list_depth, self.label_list_depth = get_sframe_paths_labels(cefa_dataset_depth, phase=mode, ratio=1)
        self.image_list_ir, self.label_list_ir = get_sframe_paths_labels(cefa_dataset_ir, phase=mode, ratio=1)


        self.miss_modal = miss_modal
        self.fill = fill
        self.transform = transform

        print(len(self.image_list_rgb), sum(np.array(self.label_list_rgb)))

    def __getitem__(self, idx):
        image_path_rgb = self.image_list_rgb[idx]
        label_rgb = self.label_list_rgb[idx]

        image_path_depth = self.image_list_depth[idx]
        label_depth = self.label_list_depth[idx]

        # print(image_path_rgb, image_path_depth)
        print(label_rgb, label_depth)

        image_rgb = cv2.imread(image_path_rgb)
        image_depth = cv2.imread(image_path_depth)


        sample = {'image_x': image_rgb,'image_depth': image_depth, 'binary_label': label_depth}

        if self.transform:
            sample = self.transform(sample)

        # print(sample)
        return sample


    def __len__(self):
        return len(self.label_list_rgb)
