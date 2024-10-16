import torch
import torch.nn.functional as F
import numpy
import random
import os
import math
import warnings
import librosa
import pathlib
import cv2
import python_speech_features
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import S
import pdb

# data = [
#     ("Actor_01", "angry", (0, 95)),
#     ("Actor_01", "disgusted", (93, 188)),
#     ("Actor_01", "fear", (186, 281)),
#     ("Actor_01", "happy", (279, 374)),
#     ("Actor_01", "neutral", (372, 467)),
#     ("Actor_01", "sad", (465, 560)),
#     ("Actor_01", "surprised", (558, 653)),
#     ("Actor_02", "angry", (0, 108)),
#     ("Actor_02", "disgusted", (106, 214)),
#     ("Actor_02", "fear", (212, 320)),
#     ("Actor_02", "happy", (318, 426)),
#     ("Actor_02", "neutral", (424, 532)),
#     ("Actor_02", "sad", (530, 638)),
#     ("Actor_02", "surprised", (636, 744)),
#     ("Actor_03", "angry", (0, 104)),
#     ("Actor_03", "disgusted", (102, 206)),
#     ("Actor_03", "fear", (204, 308)),
#     ("Actor_03", "happy", (306, 410)),
#     ("Actor_03", "neutral", (408, 512)),
#     ("Actor_03", "sad", (510, 614)),
#     ("Actor_03", "surprised", (612, 716)),
#     ("Actor_04", "angry", (0, 97)),
#     ("Actor_04", "disgusted", (95, 192)),
#     ("Actor_04", "fear", (190, 287)),
#     ("Actor_04", "happy", (285, 382)),
#     ("Actor_04", "neutral", (380, 477)),
#     ("Actor_04", "sad", (475, 572)),
#     ("Actor_04", "surprised", (570, 667)),
#     ("Actor_05", "angry", (0, 108)),
#     ("Actor_05", "disgusted", (106, 214)),
#     ("Actor_05", "fear", (212, 320)),
#     ("Actor_05", "happy", (318, 426)),
#     ("Actor_05", "neutral", (424, 532)),
#     ("Actor_05", "sad", (530, 638)),
#     ("Actor_05", "surprised", (636, 744)),
#     ("Actor_06", "angry", (0, 103)),
#     ("Actor_06", "disgusted", (101, 204)),
#     ("Actor_06", "fear", (202, 305)),
#     ("Actor_06", "happy", (303, 406)),
#     ("Actor_06", "neutral", (404, 507)),
#     ("Actor_06", "sad", (505, 608)),
#     ("Actor_06", "surprised", (606, 709))
# ]

data = [
    ("Actor_01", "angry", (0, 133)),
    ("Actor_01", "disgusted", (131, 262)),
    ("Actor_01", "fear", (260, 377)),
    ("Actor_01", "happy", (375, 493)),
    ("Actor_01", "neutral", (491, 586)),
    ("Actor_01", "sad", (584, 696)),
    ("Actor_01", "surprised", (694, 792)),
    ("Actor_02", "angry", (0, 120)),
    ("Actor_02", "disgusted", (118, 239)),
    ("Actor_02", "fear", (237, 345)),
    ("Actor_02", "happy", (343, 459)),
    ("Actor_02", "neutral", (457, 565)),
    ("Actor_02", "sad", (563, 673)),
    ("Actor_02", "surprised", (671, 785)),
    ("Actor_03", "angry", (0, 135)),
    ("Actor_03", "disgusted", (133, 251)),
    ("Actor_03", "fear", (249, 352)),
    ("Actor_03", "happy", (350, 465)),
    ("Actor_03", "neutral", (463, 567)),
    ("Actor_03", "sad", (565, 676)),
    ("Actor_03", "surprised", (674, 771)),
    ("Actor_04", "angry", (0, 115)),
    ("Actor_04", "disgusted", (113, 231)),
    ("Actor_04", "fear", (229, 336)),
    ("Actor_04", "happy", (334, 447)),
    ("Actor_04", "neutral", (445, 542)),
    ("Actor_04", "sad", (540, 653)),
    ("Actor_04", "surprised", (651, 757)),
    ("Actor_05", "angry", (0, 109)),
    ("Actor_05", "disgusted", (107, 243)),
    ("Actor_05", "fear", (241, 334)),
    ("Actor_05", "happy", (332, 434)),
    ("Actor_05", "neutral", (432, 540)),
    ("Actor_05", "sad", (538, 632)),
    ("Actor_05", "surprised", (630, 736)),
    ("Actor_06", "angry", (0, 134)),
    ("Actor_06", "disgusted", (132, 251)),
    ("Actor_06", "fear", (249, 361)),
    ("Actor_06", "happy", (359, 467)),
    ("Actor_06", "neutral", (465, 568)),
    ("Actor_06", "sad", (566, 687)),
    ("Actor_06", "surprised", (685, 781))
]
# 创建一个空字典来存储数据
inter_id = {}

# 遍历数据列表，并将数据添加到字典中
for item in data :
    session, emotion, time_range = item
    if session not in inter_id:
        inter_id[session] = {}
    inter_id[session][emotion] = time_range
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str,
                    help=('Paths to the generated images'))
parser.add_argument('--data_dir', type=str,
                    help=('test data dir containing _frame_info.txt and audios'))
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed, default: 1')
parser.add_argument('--nframe', type=int, default=5,
                    help='num_frames in one clip')
parser.add_argument('--skip_frame', type=int, default=0,
                    help='skip first n frames')
parser.add_argument('--margin', type=int, default=0,
                    help='face image margin')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--actors', nargs='+', default=['M003', 'M009', 'W029'])


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
audio_postfix = '.wav' # '.m4a'
sample_rate = 16000

def load_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio


def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists


class SyncNetInstance(torch.nn.Module):

    def __init__(self, device, pretrain_weight='metrics/syncnet_v2.model', num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__()
        self.device = device
        self.model = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(device)
        self.model.load_state_dict(torch.load(pretrain_weight, device))
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, image_list, audio_file, fps=30, batch_size=64, nframe=5, skip_frame=0, margin=70):

        images = []

        for fname in image_list:
            img_input = cv2.imread(fname)
            h, w, _ = img_input.shape
            m = margin // 2
            img_input = img_input[m:h-m, m:w-m, :]
            img_input = cv2.resize(img_input, (224,224))
            images.append(img_input)

        im = numpy.expand_dims(numpy.stack(images,axis=0), axis=0) # 1,n,h,w,c
        im = numpy.transpose(im, (0,4,1,2,3)) # 1,c,n,h,w
        imtv = torch.from_numpy(im.astype(numpy.float32))

        # ========== ==========
        # Load audio
        # ========== ==========
        frame_len = sample_rate // fps
        audio = load_audio(audio_file)[frame_len*skip_frame:]

        mfcc = python_speech_features.mfcc(audio, sample_rate, winstep=1/(fps*4)) # n,13
        mfcc = numpy.transpose(mfcc, (1, 0)) # 13,n
        mfcc = numpy.expand_dims(mfcc, axis=0) # 1,13,n
        mfcc = numpy.expand_dims(mfcc, axis=0) # 1,1,13,n
        cct = torch.from_numpy(mfcc.astype(numpy.float32))

        lastframe = min(len(images)-nframe, math.floor(mfcc.shape[-1]//4)-nframe)

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========
        im_feat = []
        cc_feat = []

        for i in range(0, lastframe, batch_size):
            im_batch = [ imtv[:, :, j:j+nframe, :, :] for j in range(i,min(lastframe,i+batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.model.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.cpu())

            cc_batch = [ cct[:, :, :, j*4:(j+nframe)*4] for j in range(i,min(lastframe,i+batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out = self.model.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        dists = calc_pdist(im_feat, cc_feat)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)
        conf = torch.median(mdist) - minval

        return conf, minval


def compute_statistics_of_path(path, data_dir, actor, model, skip_frame=0, margin=70, verbose=False):
    path = pathlib.Path(path)
    files = sorted([str(file) for ext in IMAGE_EXTENSIONS
                    for file in path.glob('**/*.{}'.format(ext))])
    files = list(filter(lambda x: actor in x, files))

    get_idx = lambda x : int(os.path.basename(x).rsplit('.', 1)[0].split('_')[0])
    start_idx = get_idx(files[0])
    end_idx = get_idx(files[-1])
    with open(os.path.join(data_dir, actor, 'videos/_frame_info.txt')) as f:
        frame_infos = f.read().splitlines()

    videos = {}
    em = files[0].split("/")[-3]
    idx = inter_id[actor][em]
    start_idx = idx[0]
    end_idx = idx[1]
    for i, line in enumerate(frame_infos[start_idx:end_idx]):
        if i > (len(files)-1):
            break
        video_name = line.split(' ')[0].rsplit('_', 1)[0]
        if video_name in videos:
            videos[video_name].append(files[i])
        else:
            videos[video_name] = [files[i]]
    
    dists = []
    confs = []
    for v, img_list in videos.items():
        audio_file = os.path.join(data_dir, actor, 'audios', v+audio_postfix)

        if not os.path.exists(audio_file):
            continue
        try:
            conf, dist = model.evaluate(img_list, audio_file, skip_frame=skip_frame, margin=margin)
            dists.append(dist)
            confs.append(conf)
        except:
            pass

    return dists, confs


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = SyncNetInstance(args.device)
    dists = []
    confs = []
    for actor in args.actors:
        dists_, confs_ = compute_statistics_of_path(args.path, args.data_dir, actor, model, args.skip_frame, args.margin)
        dists += dists_
        confs += confs_
    mdist = torch.stack(dists).mean().item()
    mconf = torch.stack(confs).mean().item()
    print('LSE-D: %.3f' % mdist, end=' ')
    #print('LSE-C: ', mconf)

