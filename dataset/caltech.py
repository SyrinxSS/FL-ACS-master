from PIL import Image
import os
import os.path

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torch.utils.data import Dataset

import numpy as np


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", transform=None,
                 target_transform=None, download=False):
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index])))

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9")
        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
            self.root,
            filename="101_Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91")

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)


class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        super(Caltech256, self).__init__(os.path.join(root, 'caltech256'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            image_paths = os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
            image_paths = [path for path in image_paths if '.jpg' in path]
            n = len(image_paths)
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index])))

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d")


class TransformedDataset(Dataset):
    def __init__(self,
                 ds,
                 transform_default=None,
                 transform_strong=None,
                 return_weight=True,
                 return_index=False):
        self.transform_default = transform_default
        self.transform_strong = transform_strong
        self.ds = ds
        self.return_weight = return_weight
        self.augment_subset = []
        self.return_index = return_index
        # self.subset = []
        # Initialize subset
        self.set_subset(np.arange(len(ds)))


    def set_subset(self, idxs, subset_weights=None):
        if subset_weights is None:
            subset_weights = np.array([1 for _ in range(len(idxs))])

        assert np.all(idxs < len(self.ds))
        assert len(subset_weights) == len(idxs)

        self.subset = idxs
        self.subset_weights = subset_weights

    def set_augment_subset(self, idxs, augment_subset_weights=None):
        print("set_augment_subset")
        if augment_subset_weights is None:
            augment_subset_weights = np.array([1 for _ in range(len(idxs))])

        # assert set(idxs).issubset(set(self.subset))
        assert np.all(idxs < len(self.ds))
        assert len(augment_subset_weights) == len(idxs)

        self.augment_subset = idxs
        self.augment_subset_weights = augment_subset_weights


    def __len__(self):
        return len(self.subset) + len(self.augment_subset)

    def __getitem__(self, idx):
        if idx < len(self.subset):
            # print("Non-core")
            weight = self.subset_weights[idx]
            ds_idx = self.subset[idx]
            #print(self.ds[ds_idx])
            sample, label = self.ds[ds_idx]
            if self.transform_default:
                sample = self.transform_default(sample)
                if sample.shape[0] == 1:
                    sample = sample.repeat(3,1,1)
        else:
            # print("Core")
            weight = self.augment_subset_weights[idx - len(self.subset)]
            ds_idx = self.augment_subset[idx - len(self.subset)]
            sample, label = self.ds[ds_idx]
            if self.transform_strong:
                # print(self.transform_strong)
                sample = self.transform_strong(sample)
                if sample.shape[0] == 1:
                    sample = sample.repeat(3,1,1)

        if self.return_weight:
            if self.return_index:
                return sample, label, weight, ds_idx
            else:
                return sample, label, weight
        elif self.return_index:
            return sample, label, idx
        else:
            return sample, label


def get_caltech_datasets(root="./data/"):
    NUM_TRAINING_SAMPLES_PER_CLASS = 60
    ds = Caltech256(root)

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]
    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))
    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    return train_set, test_set

