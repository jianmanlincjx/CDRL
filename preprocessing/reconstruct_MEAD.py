import os, sys
import numpy as np
from time import time
import argparse
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F
import torchvision
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DECA.decalib.deca import DECA
from DECA.decalib.datasets import datasets
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg

VID_EXTENSIONS = ['.mp4']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)


def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main():
    print('---------- 3D face reconstruction (DECA) on MEAD database --------- \n')    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--root', type=str, default='/data3/JM/MEAD', help='Path to MEAD database.')
    parser.add_argument('--actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003','M009','W029'])
    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Print Arguments
    print_args(parser, args)

    deca_cfg.model.use_tex = True
    deca = DECA(config = deca_cfg, device=device)

    for j, actor in enumerate(args.actors):
        if os.path.exists(os.path.join(args.root, actor + '_deca.pkl')):
            print(f'{j+1}/{len(args.actors)}: {actor} already processed')
        else:
            # find videos to be processed; we only need frontal videos with maximum intensity of emotion
            video_files = []
            actor_path = os.path.join(args.root, actor, "video")
            for root, _, fnames in sorted(os.walk(actor_path)):
                for fname in sorted(fnames):
                    if 'front' in root and is_video_file(fname):
                        if 'neutral' in root:
                            video_files.append(os.path.join(root, fname))
                        else:
                            if not 'contempt' in root and 'level_3' in root:
                                video_files.append(os.path.join(root, fname))

            actor_data = []
            # perform 3d reconstruction for every video
            for video_file in video_files:
                emotion = video_file.split('/')[-3]
                dataset = datasets.TestData(video_file, iscrop=True, face_detector='fan', scale=1.25, device=device)
                # run DECA
                print(f'Reconstructing faces and saving results for {video_file}')
                params = []
                for i in tqdm(range(len(dataset))):
                    images = dataset[i]['image'].to(device)[None,...]
                    import cv2
                    image_path = "/data2/JM/code/NED-main/render_train/M003/faces/000000/000001.png"
                    images = torch.from_numpy(cv2.resize(cv2.imread(image_path), (224, 224))).float().permute(2, 0, 1)[None, ...].cuda()
                    codedict, opdict, visdict = deca.run(image_path)
                    with open("MEAD_data/Z_Training_Aids/000000_000001.pkl", "wb") as f:
                        pickle.dump(codedict, f)
                    exit()
                    # x = F.grid_sample(opdict['uv_texture'], opdict['grid'].detach(), align_corners=False).squeeze(0)
                    # torchvision.utils.save_image(x, "xxx.png", normalize=True)
                    with torch.no_grad():
                        codedict = deca.encode(images)
                        opdict_reference, visdict_reference = deca.decode(codedict)
                        x = F.grid_sample(opdict_reference['uv_texture'], opdict_reference['grid'].detach(), align_corners=False).squeeze(0)
                        torchvision.utils.save_image(x, "xxx.png", normalize=True)
                        # with open("MEAD_data/Z_Training_Aids/000002_000120.pkl", "wb") as f:
                        #     pickle.dump(codedict, f)
                        exit()
                        params.append(np.concatenate((codedict['pose'].cpu().numpy()[:,3:], codedict['exp'].cpu().numpy()), 1))   # jaw + expression params

                actor_data.append((np.concatenate(params, 0), emotion))
            results_pth = os.path.join(args.root, actor + '_deca.pkl')
            with open(results_pth, "wb") as f:
                pickle.dump(actor_data, f)
            print(f'{j+1}/{len(args.actors)}: {actor} processed')
    print('DONE!')


if __name__=='__main__':
    main()
