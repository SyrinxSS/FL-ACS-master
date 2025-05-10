import torchvision
import torch
import torch.nn as nn
import time
import torchvision.models as models
from models.resnet_imagenet import ResnetWrapper
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
import models.resnet as resnet
from models.vgg import VGG
import torch.nn.functional as F
from models.wide_resnet import WideResNet
from tqdm import tqdm

# from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights

resnet_model_names = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(arch, CLASS_NUM, pretrained=False, feature_extract=True):
    if arch == 'resnet50':
        if pretrained:
            model = torchvision.models.resnet50(pretrained=pretrained)
            # model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, CLASS_NUM)
            model = torch.nn.DataParallel(ResnetWrapper(model))
        else:
            model = torch.nn.DataParallel(
                ResnetWrapper(torchvision.models.resnet50(num_classes=CLASS_NUM, pretrained=pretrained)))
    elif arch == 'resnet18':
        print("ResNet-18")
        if pretrained:
            model = torchvision.models.resnet18(pretrained=pretrained)
            # model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, CLASS_NUM)
            model = torch.nn.DataParallel(ResnetWrapper(model))
        else:
            model = torch.nn.DataParallel(
                ResnetWrapper(torchvision.models.resnet18(num_classes=CLASS_NUM, pretrained=pretrained)))
    elif arch == 'resnet101':
        if pretrained:
            model = torchvision.models.resnet101(pretrained=pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, CLASS_NUM)
            model = torch.nn.DataParallel(ResnetWrapper(model))
        else:
            model = torch.nn.DataParallel(
                ResnetWrapper(torchvision.models.resnet101(num_classes=CLASS_NUM, pretrained=pretrained)))
    elif arch == 'vgg19_cifar':
        '''if pretrained:
            model = torchvision.models.vgg19(pretrained=pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, CLASS_NUM)

        else:'''
        # model = torch.nn.DataParallel(VGG(img_size=32, num_class=CLASS_NUM))
        model = VGG(img_size=32, num_class=CLASS_NUM)
    elif arch == 'vgg19':
        # model = torch.nn.DataParallel(VGG(img_size=224, num_class=CLASS_NUM))
        model = VGG(img_size=224, num_class=CLASS_NUM)
    elif arch == 'w28_10':
        model = torch.nn.DataParallel(WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=CLASS_NUM))
    elif arch == 'w40_2':
        model = torch.nn.DataParallel(WideResNet(depth=40, widen_factor=2, dropout_rate=0.0, num_classes=CLASS_NUM))
    elif arch == 'w28_2':
        model = torch.nn.DataParallel(WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=CLASS_NUM))
    elif arch in resnet_model_names:
        model = torch.nn.DataParallel(resnet.__dict__[arch](CLASS_NUM))
    model.cuda()
    return model


# def train(train_loader, model, criterion, optimizer, half=False, return_loss=False):
def train(train_loader, model, optimizer, half=False, return_loss=False, use_weight=False):
    """
        Run one train epoch
    """
    print(f"Training - {len(train_loader.dataset)} examples")
    print("Current Learning Rate: {}".format(optimizer.param_groups[0]['lr']))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_ema = 0.0
    # switch to train mode
    model.train()

    end = time.time()
    if use_weight:
        print("User weight.")
        for input, target, weight in tqdm(train_loader):

            # 使用torch.eq()检查每个元素是否等于1，然后使用torch.all()检查所有元素
            # all_ones = torch.all(torch.eq(weight, torch.ones_like(weight)))

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            weight = weight.cuda()
            target_var = target
            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = F.cross_entropy(output, target_var)
            loss = (loss * weight).mean()  # (Note)
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1

            # compute gradient and do SGD step
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if return_loss:
            return data_time.sum, batch_time.sum, losses.avg
        else:
            return data_time.sum, batch_time.sum
    else:
        for input, target, _ in tqdm(train_loader):

            # 使用torch.eq()检查每个元素是否等于1，然后使用torch.all()检查所有元素
            # all_ones = torch.all(torch.eq(weight, torch.ones_like(weight)))

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = F.cross_entropy(output, target_var)
            # loss = (loss * weight).mean()  # (Note)
            # loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            # compute gradient and do SGD step
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if return_loss:
            return data_time.sum, batch_time.sum, losses.avg
        else:
            return data_time.sum, batch_time.sum


# def validate(val_loader, model, criterion, weights_per_class=None, half=False):
def validate(val_loader, model, weights_per_class=None, half=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for input, target in tqdm(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = F.cross_entropy(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, weights_per_class=weights_per_class)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), weights_per_class=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if weights_per_class is None:
            correct_k = correct[:k].view(-1).float().sum(0)
        else:
            weights = weights_per_class[target.cpu()]
            correct_k = (correct[:k].view(-1).float().cpu() * weights).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate_per_class(val_loader, model, classes, weights_per_class=None, half=False):
    """
    Run evaluation and compute accuracy for each class.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Initialize counters for each class
    num_classes = len(
        weights_per_class) if weights_per_class is not None else classes  # Default to 10 classes if not specified
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for input, target in tqdm(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = F.cross_entropy(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, class_correct_batch, class_total_batch = accuracy_per_class(output.data, target, topk=(1,),
                                                                               weights_per_class=weights_per_class,
                                                                               classes=num_classes)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Update class-specific counters
            for i in range(num_classes):
                class_correct[i] += class_correct_batch[i]
                class_total[i] += class_total_batch[i]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Compute accuracy for each class
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

    return top1.avg, losses.avg, class_accuracy


def accuracy_per_class(output, target, classes, topk=(1,), weights_per_class=None):
    """Computes the precision@k for the specified values of k and returns class-wise correct predictions"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    class_correct = [0] * len(weights_per_class) if weights_per_class is not None else [0] * classes
    class_total = [0] * len(weights_per_class) if weights_per_class is not None else [0] * classes

    for k in topk:
        if weights_per_class is None:
            correct_k = correct[:k].view(-1).float().sum(0)
        else:
            weights = weights_per_class[target.cpu()]
            correct_k = (correct[:k].view(-1).float().cpu() * weights).sum(0)

        # Update class-specific counters
        for i in range(batch_size):
            label = target[i].item()
            class_total[label] += 1
            class_correct[label] += correct[:k].view(-1)[i].item()

        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], class_correct, class_total