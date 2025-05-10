# from random import random
import random
import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
from dataset.caltech import Caltech256
from dataset.tinyimagenet import TinyImageNet
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .folder import ImageFolder

from .utils import check_integrity, download_and_extract_archive
from .vision import VisionDataset


class CIFAR10(VisionDataset):
    """
    Adapted version of CIFAR10 dataset
    """
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return index, img, target
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"

        return f"Split: {split}"


class CIFAR10Corrupt(VisionDataset):
    base_folder = "cifar-10-corrupt"

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        img_path = os.path.join(root, self.base_folder, "img.bin")
        target_path = os.path.join(root, self.base_folder, "targets.bin")
        with open(img_path, "rb") as f1:
            self.data = pickle.load(f1)
        with open(target_path, "rb") as f2:
            self.targets = pickle.load(f2)
        self.classes = datasets.CIFAR10(root, train=True, download=True, transform=transform).classes

    def _check_integrity(self) -> bool:
        # 重写 _check_integrity 方法，使其总是返回 True
        return True

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class CIFAR100Core(CIFAR100):
    """
        Coreset constructed from CIFAR-100
    """

    def __init__(self, drop_id, **kwargs):
        # drop_id must be a list containing index to drop
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)


class CIFAR100Corrupt(VisionDataset):
    base_folder = "cifar-100-corrupt"

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        img_path = os.path.join(root, self.base_folder, "img.bin")
        target_path = os.path.join(root, self.base_folder, "targets.bin")

        with open(img_path, "rb") as f1:
            self.data = pickle.load(f1)
        with open(target_path, "rb") as f2:
            self.targets = pickle.load(f2)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
        # return index, img, target

    def __len__(self):
        return len(self.data)


class CIFAR100CorruptCore(CIFAR100Corrupt):

    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)


class CIFAR10Noisy(CIFAR10):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        label_path = os.path.join(root, "noisy/cifar10.npy")
        self.targets = np.load(label_path)


class CIFAR10Corrupt(VisionDataset):
    base_folder = "cifar-10-corrupt"

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        img_path = os.path.join(root, self.base_folder, "img.bin")
        target_path = os.path.join(root, self.base_folder, "targets.bin")

        with open(img_path, "rb") as f1:
            self.data = pickle.load(f1)
        with open(target_path, "rb") as f2:
            self.targets = pickle.load(f2)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
        # return index, img, target

    def __len__(self):
        return len(self.data)


class CIFAR100Noisy(CIFAR100):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        label_path = os.path.join(root, "noisy/cifar100.npy")
        self.targets = np.load(label_path)


class CIFAR100NoisyCore(CIFAR100Noisy):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.delete(self.targets, drop_id, axis=0)


class CIFAR100Attack(CIFAR100):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = np.load("data/attack/cifar_train.npy")

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)

        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return index, img, label


class CIFAR100AttackCore(CIFAR100Attack):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class TinyNoisy(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(loader=pil_loader, **kwargs)
        self.data = np.load("data/noisy/tiny_img.npy")
        self.target = np.load("data/noisy/tiny_target.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class TinyAttack(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(loader=pil_loader, **kwargs)

        self.data = np.load("data/attack/tiny_img.npy")
        self.target = np.load("data/attack/tiny_target.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        return index, sample, target


class TinyAttackCore(TinyAttack):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        drop_id = np.sort(drop_id)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.target = np.array(self.target)
        self.target = np.delete(self.target, drop_id, axis=0)


class TinyNoisyCore(TinyNoisy):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        drop_id = np.sort(drop_id)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.target = np.array(self.target)
        self.target = np.delete(self.target, drop_id, axis=0)


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, imb_factor_second=None):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.cls_num = 10

        self.img_num_list = self.get_img_num_per_cls(imb_type, imb_factor, imb_factor_second)
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, imb_type, imb_factor, imb_factor_second=None):
        img_max = len(self.data) / self.cls_num
        if imb_type == 'binary_step' or imb_type == 'fixed' or imb_type == 'binary':
            self.cls_num = 2
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                print("Class: {}, Number of samples: {}".format(cls_idx, num))
                img_num_per_cls.append(int(num))
        elif imb_type == 'fixed':
            first_num = imb_factor
            second_num = imb_factor_second
            img_num_per_cls.append(int(img_max * first_num))
            img_num_per_cls.append(int(img_max * second_num))
        elif imb_type == 'binary_step':
            second_num = img_max * (imb_factor ** (1 / (self.cls_num - 1.0)))
            first_num = img_max - second_num
            img_num_per_cls.append(int(first_num))
            img_num_per_cls.append(int(second_num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:

            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, imb_factor_second=None):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.cls_num = 100

        self.img_num_list = self.get_img_num_per_cls(imb_type, imb_factor, imb_factor_second)
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, imb_type, imb_factor, imb_factor_second=None):
        img_max = len(self.data) / self.cls_num
        if imb_type == 'binary_step' or imb_type == 'fixed' or imb_type == 'binary':
            self.cls_num = 2
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                print("Class: {}, Number of samples: {}".format(cls_idx, num))
                img_num_per_cls.append(int(num))
        elif imb_type == 'fixed':
            first_num = imb_factor
            second_num = imb_factor_second
            img_num_per_cls.append(int(img_max * first_num))
            img_num_per_cls.append(int(img_max * second_num))
        elif imb_type == 'binary_step':
            second_num = img_max * (imb_factor ** (1 / (self.cls_num - 1.0)))
            first_num = img_max - second_num
            img_num_per_cls.append(int(first_num))
            img_num_per_cls.append(int(second_num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:

            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


def get_caltech_datasets(root="./data/", return_extra_train=True):
    # 每个类别用于训练的样本数量
    NUM_TRAINING_SAMPLES_PER_CLASS = 60
    ds = Caltech256(root)
    # 每个类别的起始索引
    class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]
    # 生成训练集的索引
    train_indices = sum(
        [np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],
        [])
    # 生成测试集的索引
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))
    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)
    # 创建一个额外的训练集对象，该对象与train_set相同
    train_set_2 = Subset(ds, train_indices)
    # 提取测试集的类别标签
    class_targets = np.array([ds.y[idx] for idx in test_indices])
    counts = np.unique(class_targets, return_counts=True)[1]
    # 计算类别权重，这里使用的是平衡权重的倒数，目的是在损失函数中对不平衡的类别进行调整
    class_weights = counts.sum() / (counts * len(counts))
    class_weights = torch.Tensor(class_weights)
    # 训练集的类别标签
    train_labels = np.array([ds.y[idx] for idx in train_indices])

    if return_extra_train:
        # 返回一个与训练集完全相同的数据集
        return train_set, test_set, train_set_2, class_weights, train_labels
    else:
        return train_set, test_set, class_weights, train_labels


def get_tinyimagenet_datasets(root="./data/tiny-imagenet-200", return_extra_train=True):
    train_set = TinyImageNet(root, split='train')
    test_set = TinyImageNet(root, split='val')
    train_set_2 = TinyImageNet(root, split='train')
    train_labels = train_set.y

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_tinyimagenet_Noisy(root="./data/tiny-imagenet-200", return_extra_train=True):
    train_set = TinyNoisy(root=root)
    test_set = TinyImageNet(root, split='val')
    train_set_2 = TinyNoisy(root=root)
    train_labels = train_set.targets

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_imagenet_datasets(root="./data/imagenet", return_extra_train=True):
    train_set = torchvision.datasets.ImageNet(root, split='train')
    test_set = torchvision.datasets.ImageNet(root, split='val')
    train_set_2 = torchvision.datasets.ImageNet(root, split='train')
    train_labels = train_set.targets

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_cifar10_datasets(root="./data/cifar-10", return_extra_train=True):
    train_set = torchvision.datasets.CIFAR10(root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=True)
    train_set_2 = torchvision.datasets.CIFAR10(root, train=True, download=True)
    train_labels = train_set.targets

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_cifar100_datasets(root="./data/cifar-100", return_extra_train=True):
    train_set = torchvision.datasets.CIFAR100(root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root, train=False, download=True)
    train_set_2 = torchvision.datasets.CIFAR100(root, train=True, download=True)
    train_labels = train_set.targets

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_all_svhn_datasets(root="./data/svhn", return_extra_train=True):
    train_set = torchvision.datasets.SVHN(root, split='train', download=True)
    extra_set = torchvision.datasets.SVHN(root, split='extra', download=True)
    total_train_set = torch.utils.data.ConcatDataset([train_set, extra_set])

    test_set = torchvision.datasets.SVHN(root, split='test', download=True)

    train_set_2 = torchvision.datasets.SVHN(root, split='train', download=True)
    extra_set_2 = torchvision.datasets.SVHN(root, split='extra', download=True)
    total_train_set_2 = torch.utils.data.ConcatDataset([train_set_2, extra_set_2])
    # 将 NumPy 数组转换为 PyTorch 张量
    train_labels = torch.tensor(train_set.labels)  # 将 train_set 的标签转换为张量
    extra_labels = torch.tensor(extra_set.labels)  # 将 extra_set 的标签转换为张量

    # 连接两个张量
    total_train_labels = torch.cat((train_labels, extra_labels))

    if return_extra_train:
        return total_train_set, test_set, total_train_set_2, None, train_labels
    else:
        return total_train_set, test_set, None, train_labels


def get_core_svhn_datasets(root="./data/svhn", return_extra_train=True):
    train_set = torchvision.datasets.SVHN(root, split='train', download=True)
    test_set = torchvision.datasets.SVHN(root, split='test', download=True)
    train_set_2 = torchvision.datasets.SVHN(root, split='train', download=True)
    train_labels = train_set.labels

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR10_imbalanced_datasets(root="./data/cifar-10", return_extra_train=True):
    train_set = IMBALANCECIFAR10(root=root, train=True, download=True)
    test_set = IMBALANCECIFAR10(root=root, train=False, download=True)
    train_set_2 = IMBALANCECIFAR10(root=root, train=True, download=True)
    train_labels = train_set.targets
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR100_imbalanced_datasets(root="./data/cifar-100", return_extra_train=True):
    train_set = IMBALANCECIFAR100(root=root, train=True, download=True)
    test_set = IMBALANCECIFAR100(root=root, train=False, download=True)
    train_set_2 = IMBALANCECIFAR100(root=root, train=True, download=True)
    train_labels = train_set.targets
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR100Corrupt(root="./data", return_extra_train=True):
    train_set = CIFAR100Corrupt(root=root)
    test_set = torchvision.datasets.CIFAR100(root="./data/cifar-100", train=False, download=True)
    train_set_2 = CIFAR100Corrupt(root=root)
    train_labels = train_set.targets
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR100Noisy(root="./data", return_extra_train=True):
    train_set = CIFAR100Noisy(root=root)
    test_set = torchvision.datasets.CIFAR100(root="./data/cifar-100", train=False, download=True)
    train_set_2 = CIFAR100Noisy(root=root)
    train_labels = train_set.targets
    total = [0] * len(list(set(train_labels)))
    for i in train_labels:
        total[i] += 1
    print(total)
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


class CIFAR10IM(CIFAR10):
    """
    CIFAR-10-IM dataset: Randomly select three classes and drop half of their training samples.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

        # 随机选择三个类
        self.selected_classes = random.sample(range(10), 3)
        print(f"Selected classes: {self.selected_classes}")

        # 舍弃掉这些类中一半的训练样本
        self._drop_half_samples()

    def _drop_half_samples(self):
        # 获取原始数据和标签
        data = self.data
        targets = self.targets

        # 对每个选中的类进行处理
        for cls in self.selected_classes:
            # 获取该类的索引
            indices = np.where(np.array(targets) == cls)[0]
            # 随机打乱索引
            np.random.shuffle(indices)
            # 舍弃掉一半的样本
            drop_indices = indices[:len(indices) // 2]
            # 更新数据和标签
            data = np.delete(data, drop_indices, axis=0)
            targets = np.delete(targets, drop_indices, axis=0)

        # 更新数据集
        self.data = data
        self.targets = targets


def get_CIFAR10IM(root="./data", return_extra_train=True):
    train_set = CIFAR10IM(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    # train_set_2 = CIFAR10IM(root=root, train=True, download=True)
    train_set_2 = train_set
    train_labels = train_set.targets
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR10Corrupt(root="./data", return_extra_train=True):
    train_set = CIFAR10Corrupt(root=root)
    test_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=False, download=True)
    train_set_2 = CIFAR10Corrupt(root=root)
    train_labels = train_set.targets
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_CIFAR10Noisy(root="./data", return_extra_train=True):
    train_set = CIFAR10Noisy(root=root)
    print(train_set.targets)
    test_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=False, download=True)
    train_set_2 = CIFAR10Noisy(root=root)
    train_labels = train_set.targets
    total = [0] * len(list(set(train_labels)))
    for i in train_labels:
        total[i] += 1
    print(total)
    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels