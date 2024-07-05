import os


train_videos = './train.csv'
test_videos = './test.csv'

train_audio_dir = '/home/data2/shicaiwei/data/audio_text_visual/vggsound/audio/train-set'
test_audio_dir = '/home/data2/shicaiwei/data/audio_text_visual/vggsound/audio/test-set'


# # test set processing
# with open(test_videos, 'r') as f:
#     files = f.readlines()
#
# for i, item in enumerate(files):
#     item=item.split(',')[0]
#     if i % 500 == 0:
#         print('*******************************************')
#         print('{}/{}'.format(i, len(files)))
#         print('*******************************************')
#     mp4_filename = os.path.join('/home/data2/shicaiwei/data/audio_text_visual/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/', item)
#     wav_filename = os.path.join(test_audio_dir, item[:-5]+'.wav')
#     if os.path.exists(wav_filename):
#         pass
#     else:
#         os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))
# #
# #
# # train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    item=item.split(',')[0]
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    mp4_filename = os.path.join('/home/data2/shicaiwei/data/audio_text_visual/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/', item)
    wav_filename = os.path.join(train_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))



#
#
