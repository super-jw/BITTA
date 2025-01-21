import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import yaml
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
from clip.custom_clip import Test_Time_Adaptation
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model
    with open(file=f"cfgs/{args.test_sets}.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    corrupt_type = args.corruption_type
    TTA_pipeline = Test_Time_Adaptation(cfg, corrupt_type, gpu)

    acc = TTA_pipeline.evaluate()

    try:
        print("=> Acc. on testset [{}-{}]: @1 {}".format(args.test_sets, corrupt_type, acc*100))
    except:
        print("=> Acc. on testset [{}]: {}".format(args.test_sets, acc*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Adaptation')
    parser.add_argument('--test_sets', type=str, default='CIFAR-10-C', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--corruption_type', type=str, default='gaussian_noise', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=0)

    main()