import copy
import csv
import os
import pickle
import librosa
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
import glob
import numpy as np
import time


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


class VGGSound(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        train_class = []
        test_class = []

        i = 0

        with open('./dataset/data/VGGSound/vggsound.csv') as f:
            csv_reader = csv.reader(f)

            for item in csv_reader:
                i += 1
                # if i>10000:
                #     break
                if i % 100 == 0:
                    print(i)

                if item[3] == 'train':
                    video_dir = os.path.join('./train_test_data/vggsound/', 'video/train-set-img',
                                             'Image-{:02d}-FPS'.format(self.args.fps),
                                             item[0] + '_' + item[1].zfill(6) + '.mp4')
                    audio_dir = os.path.join('./train_test_data/vggsound/', 'audio/train-audios/train-set',
                                             item[0] + '_' + item[1].zfill(6) + '.wav')
                    # print(video_dir)
                    # print(audio_dir)
                    # if (os.path.exists(video_dir)):
                    #     print(os.path.exists(video_dir))
                    # if (os.path.exists(audio_dir)):
                    #     print(os.path.exists(audio_dir))
                    # print(len(listdir_nohidden(video_dir))>3)
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(
                            listdir_nohidden(video_dir)) >= 3:
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: train_class.append(item[2])
                        train_label.append(item[2])
                    else:
                        print(video_dir)
                        print('\n')
                        print(audio_dir)
                        print('\n')

                if item[3] == 'test':

                    video_dir = os.path.join('./train_test_data/vggsound/', 'video/test-set-img',
                                             'Image-{:02d}-FPS'.format(self.args.fps),
                                             item[0] + '_' + item[1].zfill(6) + '.mp4')
                    audio_dir = os.path.join('./train_test_data/vggsound/', 'audio/test-audios/test-set',
                                             item[0] + '_' + item[1].zfill(6) + '.wav')
                    # if (os.path.exists(video_dir)):
                    #     print(os.path.exists(video_dir))
                    # if (os.path.exists(audio_dir)):
                    #     print(os.path.exists(audio_dir))
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(
                            listdir_nohidden(video_dir)) >= 3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: test_class.append(item[2])
                        test_label.append(item[2])

                    else:
                        print(video_dir)
                        print('\n')
                        print(audio_dir)
                        print('\n')

        # print(train_class,test_class)
        print(len(train_class))
        print(len(test_class))
        print(train_class)
        print(test_class)
        assert len(train_class) == len(test_class)
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.audio_simple_list = []
        self.audio_rate_list = []
        self.image_smaple_list=[]

        # if self.mode == 'train':

        # for i in range(len(self.audio)):
        #     if i>100000:
        #         break
        #     print("process:",i)
        #     sample, rate = librosa.load(self.audio[i], sr=16000, mono=True)
        #     self.audio_rate_list.append(rate)
        #     self.audio_simple_list.append(sample)
        #
        #     image_samples = listdir_nohidden(self.video[i])
        #     self.image_smaple_list.append(image_samples)


    def __len__(self):
        # return 10000
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        bt = time.time()
        if self.mode =='train' and idx<-1:
            sample, rate = self.audio_simple_list[idx], self.audio_rate_list[idx]
        else:
            sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)

        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # et = time.time()
        # print("audio:", et - bt)
        # bt = time.time()

        # Visual
        if self.mode =='train' and idx<-1:
            image_samples = self.image_smaple_list[idx]
        else:
            image_samples = listdir_nohidden(self.video[idx])

        # select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        # select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[i]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue
            # bt = time.time()
            img = self.transform(img)
            # et = time.time()
            # print(et-bt)
            images[i] = img
        # et = time.time()
        # print("visual:", et - bt)
        images = torch.permute(images, (1, 0, 2, 3))
        # print(images.is_contiguous())
        # images = images.contiguous()

        # label
        label = self.label[idx]

        return spectrogram, images, label
