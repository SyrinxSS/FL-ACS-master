import os
import sys
from datetime import datetime

import PIL
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
sys.path.insert(0, '../')
sys.path.insert(0, './')
import torch
import torchvision
import torchvision.transforms as transforms
import dataset.datasets as datasets
import numpy as np
from torch.utils.data import Subset, Dataset
from train import get_model, train, validate, validate_per_class
from dataset.autoaugment import ImageNetPolicy
from dataset.caltech import TransformedDataset
import utils.subset as subsetlib
import argparse
import pandas as pd
from dataset.cutout import Cutout
from augmentation.randaugment import RandAugment, ImageNetPolicy
from augmentation import BAA


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def calculate_similarity(core_set_similarity):
    """Calculate the similarity between the last two core sets."""
    if len(core_set_similarity) >= 2:
        set1 = set(core_set_similarity[-2])
        set2 = set(core_set_similarity[-1])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    else:
        print("Not enough core sets to compute similarity.")
        return None

def adjust_R_based_on_similarity(dataset, core_set_similarity, R, similarity_threshold):
    if dataset == 'cifar10' or dataset == 'svhn_core' or dataset == 'cifar10-n' or dataset == 'cifar10-c' or dataset == 'cifar10-i':
        delta = 2
    elif dataset == 'cifar100' or dataset == 'tinyimagenet':
        delta = 5
    else:
        delta = 5
        print("Unknown dataset. Using default delta of 5.")
    if len(core_set_similarity) >= 2:
        set1 = set(core_set_similarity[-2])
        set2 = set(core_set_similarity[-1])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union != 0 else 0

        print(f"Similarity between last two core sets: {similarity:.3f}")

        if similarity > similarity_threshold:
            R += delta
            print(f"Increasing R to {R} due to high similarity.")
        else:
            R = max(5, R - delta)
            print(f"Decreasing R to {R} due to low similarity.")
    else:
        print("Not enough core sets to compute similarity. Keeping R unchanged.")

    return R


def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    intersection = len(set1.intersection(set2))

    union = len(set1.union(set2))

    similarity = intersection / union if union != 0 else 0

    return similarity

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('output_dir', type=str,
                        help='directory to output csv results')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='whether to use pretrained model')
    parser.add_argument('--cutout', dest='cutout', action='store_true',
                        help='whether to use cutout augmentation')
    parser.add_argument('--noise_only', dest='noise_only', action='store_true',
                        help='whether to use noise augmentation')
    parser.add_argument('--dataset', dest='dataset', type=str, default="cifar10", help='dataset to use')
    parser.add_argument('--arch', dest='arch', type=str, default="w40_2", help='model architecture')
    parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=0.5)
    parser.add_argument('--enable_coreset_augment_weights', dest='enable_coreset_augment_weights', action='store_true',
                        help='whether to enable weights for coresets when using subset augmentation')
    parser.add_argument('--use_linear', dest='use_linear', action='store_true', help='Linear layer for coreset gradient approximation')
    parser.add_argument('--no_equal_num', dest='no_equal_num', action='store_true',
                        help='whether to use equal num from each class in coreset selection')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', '-m', type=float, metavar='M', default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-R', '--R', dest='R', type=int, metavar='R',
                        help='interval to select subset', default=1)

    args = parser.parse_args()

    dataset = args.dataset
    R = args.R
    subset_size = args.subset_size
    enable_coreset_augment_weights = args.enable_coreset_augment_weights
    use_linear = args.use_linear
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    wd = args.weight_decay
    print("Dataset: {}, Weight Decay: {}".format(args.dataset,wd))
    bs = args.batch_size
    # pretrained = args.pretrained
    pretrained = True
    output_dir = args.output_dir
    get_core_set = 0
    core_set_similarity = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Arguments:  {args}")

    if args.dataset == "tinyimagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        TRAIN_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 200
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_tinyimagenet_datasets()

    elif args.dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        TRAIN_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                transforms.Resize(size=256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 1000
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_imagenet_datasets()

    elif args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),
            transforms.ToTensor(),
            normalize,
            Cutout(1, 16)
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_cifar10_datasets()

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_cifar100_datasets()

    elif args.dataset == 'svhn_core':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(1, 20)
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(1, 20)
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        TEST_TRANSFORMS = VAL_TRANSFORMS
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_core_svhn_datasets()

    elif args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(1, 20)
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(1, 20)
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        TEST_TRANSFORMS = VAL_TRANSFORMS
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_all_svhn_datasets()

    elif args.dataset == 'cifar10-i':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR10IM()

    elif args.dataset == 'cifar100-i':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR100_imbalanced_datasets()

    elif args.dataset == "cifar100-c":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR100Corrupt()

    elif args.dataset == "cifar100-n":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 100
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR100Noisy()

    elif args.dataset == "cifar10-c":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),

            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR10Corrupt()

    elif args.dataset == "cifar10-n":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            BAA.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        NUM_CLASSES = 10
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_CIFAR10Noisy()

    model = get_model(args.arch, NUM_CLASSES, pretrained=pretrained, feature_extract=False)

    if args.arch == 'vgg19_cifar':
        model.img_size = 32
    elif args.arch == 'vgg':
        model.img_size = 224
    # model = torch.nn.DataParallel(model)

    train_set = TransformedDataset(train_set, transform_default=TRAIN_TRANSFORMS, transform_strong=TRAIN_TRANSFORMS_STRONG)
    print(train_set.transform_strong)
    test_set = TransformedDataset(test_set, transform_default=VAL_TRANSFORMS, return_weight=False)
    train_set_indexed = TransformedDataset(train_set_indexed, transform_default=VAL_TRANSFORMS, return_weight=False, return_index=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=8)
    train_indexed_loader = torch.utils.data.DataLoader(
        train_set_indexed, batch_size=bs, shuffle=True, num_workers=8)

    print(f"Params: {sum(param.numel() for param in model.parameters())}    "
    f"Trainable params: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay = wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs,
            1,  # lr_lambda computes multiplicative factor
            1e-6 / lr))
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    val_criterion = torch.nn.CrossEntropyLoss().cuda()

    N = len(train_set)
    B = int(N * subset_size)

    print("No equal num: ", args.no_equal_num)
    # no_equal_num = args.no_equal_num
    no_equal_num = False
    best_prec = 0
    best_epoch = 0
    best_class_accuracy = None
    prec = 0
    get_subset_time = 0
    training_time = 0
    val_time = 0
    global_R = R

    model_dir = "./results"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    start = datetime.now()
    for epoch in range(epochs):
        with torch.no_grad():
            if subset_size < 1 and (epoch == 0 or epoch == R):
                print("Current Subset Size: {}.".format(subset_size))
                get_core_set += 1
                subset_start = datetime.now()
                prediction_error, preds, labels = subsetlib.grad_predictions(train_indexed_loader, model, NUM_CLASSES, use_linear=use_linear)
                subset, subset_weights, _, _, _ = subsetlib.get_coreset(
                    prediction_error,
                    labels,
                    N=len(train_indexed_loader),
                    B=int(N * subset_size),
                    num_classes=NUM_CLASSES,
                    normalize_weights=True,
                    gamma_coreset=0,
                    smtk=0,
                    st_grd=0,
                    equal_num=not no_equal_num,
                    replace=False,
                    optimizer="LazyGreedy")
                core_set_similarity.append(subset)
                if len(core_set_similarity) >= 2:
                    print("Core Set Similarity: ",calculate_similarity(core_set_similarity))
                    global_R = adjust_R_based_on_similarity(args.dataset,core_set_similarity, global_R, subset_size)
                    R += global_R
                    print("Next Coreset Selection epoch: {}".format(R))
                train_loader.dataset.subset = []
                print(train_loader.dataset.transform_strong)
                train_loader.dataset.set_augment_subset(subset, augment_subset_weights=None)
                current_subset_time = (datetime.now() - subset_start).total_seconds()
                print("Get subset Time: {:.3f}s".format(current_subset_time))
                get_subset_time += current_subset_time
            elif subset_size >= 1:
                train_loader.dataset.set_augment_subset(np.arange(N), augment_subset_weights=None)

        print(f"Epoch: {epoch}  Subset: {B} Next R: {R}", end='')
        current_start = datetime.now()
        print("Current Epoch: {}, Subset Size: {}".format(epoch, len(train_loader.dataset.subset)))
        train(train_loader, model, optimizer, use_weight=enable_coreset_augment_weights)
        current_end = (datetime.now()-current_start).total_seconds()
        training_time += current_end
        print(f"Training Time: {current_end:.3f}s")
        lr_scheduler.step()
        val_start = datetime.now()
        with torch.no_grad():
            prec, _, class_accuracy = validate_per_class(val_loader, model, NUM_CLASSES, weights_per_class=class_weights)
            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                best_class_accuracy = class_accuracy
                best_name = str(args.arch) + "-" + str(subset_size) + "-" +str(best_prec) + "-" + str(best_epoch) + '.pt'
                torch.save(model.state_dict(), os.path.join(model_dir, best_name))
            print(f"Prec: {prec:.3f}  Best prec: {best_prec:.3f} @ Epoch {best_epoch}")
        val_time += (datetime.now()-val_start).total_seconds()
    overall_time = (datetime.now()-start).total_seconds()
    print("Current B: {}, R: {}, Network: {}, Time: {:.3f}s, Accuracy: {}".format(int(subset_size * N), R, args.arch, overall_time, best_prec))
    print("Total Subset Time: {:.3f}s, Total Training Time: {:.3f}s, Total Val Time: {:.3f}s".format(get_subset_time, training_time, val_time))
    for i, acc in enumerate(best_class_accuracy):
        print(f"Class {i} Accuracy: {acc:.2f}%")
    if get_core_set > 0:
        print("Average Subset Time: {:.3f}s".format(get_subset_time/get_core_set))
    result_dict = vars(args)
    result_dict['last_prec'] = prec

    result_dict['best_prec'] = best_prec
    result_dict['best_epoch'] = best_epoch
    result_dict['get_subset_time'] = get_subset_time
    result_dict['training_time'] = training_time
    result_dict['val_time'] = val_time
    result_dict['coreset'] = len(core_set_similarity)

    df = pd.DataFrame([result_dict])
    filename = os.path.join(output_dir, "result{}.csv")
    df.to_csv(filename, index=False, header=True)

if __name__ == '__main__':
    main()
    torchvision.models.resnet101()
