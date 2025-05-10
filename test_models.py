import os
import sys
from torchvision.transforms import AutoAugmentPolicy

sys.path.insert(0, '../')
sys.path.insert(0, './')
import torch
import torchvision
import torchvision.transforms as transforms
import dataset.datasets as datasets
from torch.utils.data import Subset, Dataset
from train import get_model, validate, validate_per_class
from dataset.autoaugment import ImageNetPolicy
from dataset.caltech import TransformedDataset
import argparse
import pandas as pd
from dataset.cutout import Cutout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', dest='dataset', type=str, default="cifar10", help='dataset to use')
    parser.add_argument('--arch', dest='arch', type=str, default="w40_2", help='model architecture')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    args = parser.parse_args()
    dataset = args.dataset
    bs = args.batch_size

    if args.dataset == "caltech256":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                # normalize
            ])
        NUM_CLASSES = 257
        _, test_set, _, _, _ = datasets.get_caltech_datasets(root="./data", return_extra_train=True)

    elif args.dataset == "tinyimagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        VAL_TRANSFORMS = transforms.Compose([
                # transforms.CenterCrop(size=56),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                # transforms.Resize(size=64),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 200
        _, test_set, _, _, _ = datasets.get_tinyimagenet_datasets()

    elif args.dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 1000
        _, test_set, _, _, _ = datasets.get_imagenet_datasets()

    elif args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_cifar10_datasets()

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        _, test_set, _, _, _ = datasets.get_cifar100_datasets()

    elif args.dataset == 'svhn_core':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_core_svhn_datasets()

    elif args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_all_svhn_datasets()

    elif args.dataset == 'cifar10-i':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_CIFAR10IM()

    elif args.dataset == 'cifar100-i':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        _, test_set, _, _, _ = datasets.get_CIFAR100_imbalanced_datasets()

    elif args.dataset == "cifar100-c":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        _, test_set, _, _, _ = datasets.get_CIFAR100Corrupt()

    elif args.dataset == "cifar100-n":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        _, test_set, _, _, _ = datasets.get_CIFAR100Noisy()

    elif args.dataset == "tinyimagenet-c":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        VAL_TRANSFORMS = transforms.Compose([
            # transforms.CenterCrop(size=56),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            # transforms.Resize(size=64),
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 200
        _, test_set, _, _, _ = datasets.get_tinyimagenet_datasets()

    elif args.dataset == "tinyimagenet-n":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        VAL_TRANSFORMS = transforms.Compose([
            # transforms.CenterCrop(size=56),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            # transforms.Resize(size=64),
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 200
        _, test_set, _, _, _ = datasets.get_tinyimagenet_Noisy()

    elif args.dataset == "cifar10-c":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_CIFAR10Corrupt()

    elif args.dataset == "cifar10-n":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        _, test_set, _, _, _ = datasets.get_CIFAR10Noisy()
    print(NUM_CLASSES)
    model = get_model(args.arch, NUM_CLASSES, pretrained=False, feature_extract=False)
    print(next(model.parameters()).device)
    save_path = "./results"
    model_name = "cifar100-w40_2-0.01-24.45-198.pt"
    model_path = os.path.join(save_path, model_name)
    model.load_state_dict(torch.load(model_path))

    test_set = TransformedDataset(test_set, transform_default=VAL_TRANSFORMS, return_weight=False)
    # Dataloader
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=8)
    print(f"Params: {sum(param.numel() for param in model.parameters())}    "
          f"Trainable params: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

    with torch.no_grad():
        prec, _, class_accuracy = validate_per_class(val_loader, model, NUM_CLASSES, weights_per_class=None)
        print(f"Test Accuracy: {prec:.3f}")

# python test_models.py --dataset cifar10 --arch w40_2 -b 128
