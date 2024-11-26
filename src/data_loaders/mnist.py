
import os
import os.path
from typing import Any, Callable, Optional, Tuple, Dict
from urllib.error import URLError
from .utils import check_integrity, download_resource
from .visiondataset import VisionDataset



class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``mnist_train.csv``
            and  ``mnist_test.csv`` exist.
        train (bool, optional): If True, creates dataset from ``mnist_train.csv``,
            otherwise from ``mnist_test.csv``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        "https://pjreddie.com/media/files/",
    ]

    resources = [
        ("mnist_train.csv"),
        ("mnist_test.csv"),
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

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

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()


    def _load_data(self):
        file_path = f"mnist_{'train' if self.train else 'test'}.csv"
        data, targets = self.load_csv(os.path.join(self.raw_folder, file_path))

        return data, targets
    
    def load_csv(self, path):
        """
        Load the CSV form of MNIST data without any external library
        :param path: the path of the csv file
        :return:
            data: A list of list where each sub-list with 28x28 elements
                corresponding to the pixels in each image
            labels: A list containing labels of images
        """
        data = []
        labels = []
        with open(path, 'r') as fp:
            images = fp.readlines()
            images = [img.rstrip() for img in images]

            for img in images:
                img_as_list = img.split(',')
                y = int(img_as_list[0])  # first entry as label
                x = img_as_list[1:]
                x = [int(px) / 255 for px in x]
                data.append(x)
                labels.append(y)

        return data, labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, url))
            for url in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_resource(url, download_root=self.raw_folder, filename=filename)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    
# if __name__ == "__main__":
#     mnist_ds = MNIST('./data', train=True, download=True)
#     print('loaded')