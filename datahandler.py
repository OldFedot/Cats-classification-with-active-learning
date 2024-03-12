from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os


class Cats():
    """Class for data manipulation: store references on train/test data."""
    def __init__(self, all_cats_folder, test_csv, train_csv):
        """
        Parameters:
            all_cats_folder (str): Path to folder with all cats images.
            test_csv (str): Path to .csv file with manual labeled test samples (~10 per each class).
            train_csv (str): Path to .csv file with manual labeled train samples (~40 per class).
        """
        self.cls = {"red": 0,
                    "white": 1,
                    "black": 2,
                    "grey": 3,
                    "grey_brown_stripes": 4,
                    "white_red": 5,
                    "white_black_white_grey": 6,
                    "three_colors": 7,
                    "siam": 8}

        self.root = all_cats_folder
        self.test_cats = pd.read_csv(test_csv, index_col="fname")
        self.train_cats = pd.read_csv(train_csv, index_col="fname")
        self.all_cats = self.make_full_data_df()

    def make_full_data_df(self):
        """Create pandas dataframe to store references for all images."""
        imgs = sorted(os.listdir(self.root))
        all_cats = pd.DataFrame(index=imgs, columns=["label"])
        all_cats.drop(self.test_cats.index, axis=0, inplace=True)
        return all_cats

    def __getitem__(self, idx):
        """Return image by [] operator and index."""
        img = plt.imread(os.path.join(self.root, self.all_cats.iloc[idx].name))
        return img


class CatsDataset(Dataset):
    """Pytorch Dataset class, to prepare images to training."""
    def __init__(self, root, imgs:list, labels:list, transform=None):
        """
        Constructor of CatsDataset class.
        Parameters:
             root (str): Path to folder with all cats images.
             imgs (list): List of image names
             labels (list): List of image labels
             transform (torchvisions.transforms): image transformations
        """
        super().__init__()
        self.root = root
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        """Return image by [] operator and index, with applying corresponding transforms."""
        img = plt.imread(os.path.join(self.root, self.imgs[idx]))
        if self.transform is not None:
            img = self.transform(img)
        if self.labels == None:
            return img
        else:
            return img, int(self.labels[idx])

    def __len__(self):
        """Return len of dataset"""
        return len(self.imgs)