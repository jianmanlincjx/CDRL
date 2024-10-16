import os
import random
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d, cosine_similarity
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from metrics.models import efficient_face

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int, default=4,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_samples', type=int, default=100,
                    help='num samples to eval, default: 100')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed, default: 1')
parser.add_argument('--actors', nargs='+', default=['M003', 'M009', 'W029'])
parser.add_argument('--margin', type=int, default=70,
                    help='face image margin')
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def calculate_activation_statistics(files, model, batch_size=50, device='cpu',
                                    num_workers=1, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A torch temsor of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=model.get_transforms())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = []
    iter_ = tqdm(dataloader) if verbose else dataloader
    for batch in iter_:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.ndim == 4:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)

        pred_arr.append(pred)

    return torch.cat(pred_arr, dim=0)


def compute_statistics_of_path(path, actor, model, batch_size, device,
                               num_workers=1, verbose=True):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('**/*.{}'.format(ext))])
    files = list(filter(lambda x: actor in str(x), files))
    if len(files) > args.num_samples:
        files = random.sample(files, args.num_samples)
    embeddings = calculate_activation_statistics(files, model, batch_size,
                                                 device, num_workers, verbose)

    return embeddings


def calculate_csim_given_paths(paths, actors, batch_size, device, num_workers=1, verbose=True):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = efficient_face().to(device)

    csims = []
    for actor in actors:
        embd1 = compute_statistics_of_path(paths[0], actor, model, batch_size,
                                           device, num_workers, verbose)
        embd2 = compute_statistics_of_path(paths[1], actor, model, batch_size,
                                           device, num_workers, verbose)
  
        csims.append(cosine_similarity(embd1.unsqueeze(1), embd2.unsqueeze(0), dim=2).mean())

    return torch.stack(csims).mean().item()


def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    csim_value = calculate_csim_given_paths(args.path,
                                            args.actors,
                                            args.batch_size,
                                            device,
                                            num_workers,
                                            args.verbose)
    print('CSIM: %.3f' % csim_value, end=' ')


if __name__ == '__main__':
    args = parser.parse_args()
    main()
