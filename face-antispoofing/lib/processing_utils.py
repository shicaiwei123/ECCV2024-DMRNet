import os
import numpy as np
import torch
import csv

import random




def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    '求数据集的均值方差'
    '本质是读取一个epoch的数据进行测试,只不过把一个epoch的大小设置成了所有数据'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    train = train['image_rgb']
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def seed_torch(seed=0):
    '''在使用模型的时候用于设置随机数'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataself_hist(arr):
    '''
    统计一个arr 不同数字出现的频率
    可以看做以本身为底 的直方图统计
    :param arr:
    :return:
    '''
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v

    return result


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def replace_string(path, index, new_one):
    '''
    选择更换指定未知的字符串为某个新字符串
    :param path: 原始字符串
    : index 要修改的字符位置
    :new_one  更新的目标

    :return:
    '''
    path_split = path.split('/')
    path_split[index] = new_one
    path_new = '/'.join(path_split)
    return path_new


def save_csv(csv_path, data):
    '''
    以csv格式,增量保存数据,常用域log的保存

    :param csv_path: csv 文件地址
    :param data: 要保存数据,list和arr 都可以,但是只能是一维的
    :return:
    '''
    with open(csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close()


def save_args(args, save_path):
    if os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()


def read_csv(csv_path):
    '''
    读取csv文件的内容,并且返回
    '''
    data_list = []
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        '''item 是一个list,一个item 就是一行'''
        data_list.append(item)

    return data_list


def read_txt(txt_path):
    '''
    读取txt 文件
    :param txt_path:
    :return: txt中每行的数据,结尾用'\n'
    '''

    f = open(txt_path)
    data = f.readlines()
    for index in range(len(data)):
        data[index] = data[index][:-1]
    return data


# def generate_road_txt(data_root, relative=True):
#     '''
#     生成文件夹下所有文件的路径所有并且保存为txt
#     :return:
#     :relative: 是否使用相对路径保存
#     '''
#
#     data_path_list = get_file_list(data_root)
#     data_root_split = data_root.split('/')
#     data_root_back=data_root_split.copy()
#     data_root_back.pop()
#     txt_path = os.path.join('/'.join(data_root_back), data_root_split[-1] + ".txt")
#     fw = open(txt_path, 'a')
#
#     if not relative:
#         for path in data_path_list:
#             fw.write(path)
#             fw.write('\n')
#     else:
#         for path in data_path_list:
#             path_split = path.split('/')
#             sub_split = [item for item in path_split if item not in data_root_split]
#             path_save='/'.join(sub_split)
#             fw.write(path_save)
#             fw.write('\n')


'''
数据和数据集的一些预处理操作,包括但不限于:从视频中获取图像,人脸检测,利用原始数据集,生成满足条件的数据集.
'''


def video_to_frames(pathIn='',
                    pathOut='',
                    extract_time_interval=-1,
                    only_output_video_info=False,
                    extract_time_points=None,
                    initial_extract_time=0,
                    end_extract_time=None,
                    output_prefix='frame',
                    jpg_quality=100,
                    isColor=True):
    '''
    pathIn：视频的路径，比如：F:\python_tutorials\test.mp4
    pathOut：设定提取的图片保存在哪个文件夹下，比如：F:\python_tutorials\frames1\。如果该文件夹不存在，函数将自动创建它
    only_output_video_info：如果为True，只输出视频信息（长度、帧数和帧率），不提取图片
    extract_time_points：提取的时间点，单位为秒，为元组数据，比如，(2, 3, 5)表示只提取视频第2秒， 第3秒，第5秒图片
    initial_extract_time：提取的起始时刻，单位为秒，默认为0（即从视频最开始提取）
    end_extract_time：提取的终止时刻，单位为秒，默认为None（即视频终点）
    extract_time_interval：提取的时间间隔，单位为秒，默认为-1（即输出时间范围内的所有帧）
    output_prefix：图片的前缀名，默认为frame，图片的名称将为frame_000001.jpg、frame_000002.jpg、frame_000003.jpg......
    jpg_quality：设置图片质量，范围为0到100，默认为100（质量最佳）
    isColor：如果为False，输出的将是黑白图片
    '''

    cap = cv2.VideoCapture(pathIn)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  ##视频的帧率
    dur = n_frames / fps  ##视频的时间

    ##如果only_output_video_info=True, 只输出视频信息，不提取图片
    if only_output_video_info:
        print('only output the video information (without extract frames)::::::')
        print("Duration of the video: {} seconds".format(dur))
        print("Number of frames: {}".format(n_frames))
        print("Frames per second (FPS): {}".format(fps))

        ##提取特定时间点图片
    elif extract_time_points is not None:
        if max(extract_time_points) > dur:  ##判断时间点是否符合要求
            raise NameError('the max time point is larger than the video duration....')
        try:
            os.mkdir(pathOut)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not isColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ##转化为黑白图片
                print('Write a new frame: {}, {}th'.format(success, count + 1))
                cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                count = count + 1

    else:
        ##判断起始时间、终止时间参数是否符合要求
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        ##时间范围内的每帧图片都输出
        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

        ##判断提取时间间隔设置是否符合要求
        elif extract_time_interval > 0 and extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        ##时间范围内每隔一段时间输出一张图片
        else:
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1


def frame_to_face(frame_dir, face_dir, model_name, normal_size=None, save_mode='.jpg'):
    '''
    检测人脸，并保存
    :param frame_dir: 原始帧保留的文件夹
    :param face_dir: 人脸保留的位置
    :model_name: 人脸检测的模型
    :return:
    '''
    face_detector = FaceDection(model_name=model_name)

    for root, dirs, files in os.walk(frame_dir):

        # 当子目录为空的时候，root就是不包含子文件夹的文件夹
        if dirs == []:

            for file_name in files:
                file_path = os.path.join(root, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    print("if img is None:")
                    continue

                # 人脸检测
                if normal_size is not None:
                    img_normal = cv2.resize(img, normal_size)
                else:
                    img_normal = img
                face_img = face_detector.face_detect(img_normal)
                if face_img is None:
                    print("if face_img is None:")
                    continue

                # 获取存储路径
                frame_dir_split = frame_dir.split('/')
                file_path_split = file_path.split('/')
                file_path_split.pop()  # 去掉文件名
                sub_split = [item for item in file_path_split if item not in frame_dir_split]
                save_dir = face_dir
                for item in sub_split:
                    save_dir = os.path.join(save_dir, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 存储
                save_path = os.path.join(save_dir, file_name.split('.')[0] + save_mode)
                cv2.imwrite(save_path, face_img)


def analysis(true_right, true_wrong, false_right, false_wrong):
    false_all = false_wrong + false_right
    true_all = true_wrong + true_right
    true_class = true_right + false_wrong
    false_class = true_wrong + false_right
    all = true_right + true_wrong + false_wrong + false_right

    accuracy = (true_right + false_right) / all
    APCER = false_wrong / false_all
    BPCER = true_wrong / true_all
    ACER = (APCER + BPCER) / 2
    FAR = false_wrong / true_class
    FRR = true_wrong / false_class

    print("accuracy", accuracy, "APCER", APCER, "BPCER", BPCER, "ACER", ACER, "FAR", FAR, "FRR", FRR)


def img_preview(img_dir):
    file_list = get_file_list(img_dir)
    for file_path in file_list:
        img = cv2.imread(file_path)
        if img is None:
            continue
        img_shape = img.shape
        print("img_shape", img_shape)
        cv2.namedWindow("img_show", 0)
        cv2.imshow("img_show", img)
        cv2.waitKey(0)
