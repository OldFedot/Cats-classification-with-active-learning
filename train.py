import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from model import ConvNet
from analysis import PostProcessor


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


class TrainTest():
    """Class to perform model training and evaluation."""
    def __init__(self, model,  device, learning_rate=0.001):
        """
        Constructor of TrainTest class.
        Parameters:
            model (torch.model): Pytorch ML model.
            device (str): device at wich store the model and perform training.
            learning_rate: learning rate of optimizer.
        """
        self.device = device
        self.model = model.to(device)
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # random image transforms to avoid model overfitting
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(30)
                                              ])
        # Hook to extract 64 dims output
        self.model.fc1.register_forward_hook(self.get_features("fc1"))
        self.features = {}

        # Processing
        self.softmax = torch.nn.Softmax(dim=1)

    def train_step(self, dataloader):
        """
        Perfor training step.
        Parameters:
            dataloader (torsh.utils.data.DataLoader): dataloader with training data.
        Returns:
            total_loss (float): Sumf of losses from the all batches.
            """
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        total_loss = 0

        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            X = self.transform(X)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % dataloader.batch_size == 0:
                current = batch*dataloader.batch_size + len(X)
                print(f"loss: {loss:> 5} [{current:>5d} / {size:>5}]")
        return total_loss

    def test_step(self, dataloader):
        """
        Perform test step.
        Parameters:
            dataloader (torch.utils.data.DataLoader): dataloader object of test data.
        Returns:
            total_loss (float): sumf of losses from the all batches.
            score (float): f1_score from the model evaluation
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            true_labels = np.array([])
            pred_labels = np.array([])

            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                pred = self.softmax(pred)
                total_loss = self.loss_fn(pred, y).item()
                true_labels = np.concatenate([true_labels, y.cpu().detach().numpy()], axis=0)
                pred_labels = np.concatenate([pred_labels, pred.argmax(1).cpu().detach().numpy()], axis=0)

            score = f1_score(true_labels, pred_labels, average="macro")
            return total_loss, score

    def learning_cycle(self, train_dataloader, test_dataloader, epochs):
        """
        Perform train->test training cycles.
        Parameters:
            train_dataloader (torch.utils.data.Dataloader): dataloader for train data.
            test_dataloader (torch.utils.data.Dataloader): dataloader for test data.
            epochs (int): number of traning epochs.
        Returns:
            loss_train (np.array): train loss on every epoch.
            loss_test (np.array): test loss on every epoch.
            score (np.array): model scoring on every epoch.
        """
        loss_train = np.zeros(shape=len(range(epochs)))
        loss_test = np.zeros(shape=len(range(epochs)))
        score = np.zeros(shape=len(range(epochs)))

        for i in range(epochs):
            print(f"Epoch {i + 1}\n--------------")
            loss_train[i] = self.train_step(train_dataloader)
            loss_test[i], score[i] = self.test_step(test_dataloader)

        return loss_train, loss_test, score

    def predict(self, dataloader, num_classes=9):
        """
        Make predictions.
        Parameters:
            dataloader (torch.utils.data.Dataloader): dataloader with data to process.
            num_classes (int): number of classes in classification problem.
        Returns:
            preds (np.array): model predictions with no SoftMax activation on the last layer.
            preds_64_emb: Intermideate model output, i.e. representation of original images with 64 dimensions vector.
        """
        preds = np.zeros(shape=(0, num_classes))
        preds_64_emb = np.zeros(shape=(0, 64))
        self.features = {}

        self.model.eval()
        for i, X in enumerate(dataloader):
            X = X.to(self.device)
            pred = self.model(X)
            preds = np.concatenate([preds, pred.cpu().detach().numpy()], axis=0)
            preds_64_emb = np.concatenate([preds_64_emb, self.features["fc1"].cpu().detach().numpy()], axis=0)

        return preds, preds_64_emb

    def get_features(self, name):
        """Auxiliary function to make a hoook for extraction of 64 dimensions vector of image represenation."""
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook


def main():
    """ Function to run trainin, evaluation and manual labeling of new data."""
    # Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device to perform training
    batch_size = 10 # Batch size
    new_size = 128 # Update image size. Original size is 256, so we reduced it twice
    epochs = 30 # Number of training epochs for single active learning iteration
    n_worst = 30 # Number of worst classified images for manual labeling at each active learning iteration

    # Image pre-processing
    img_transform = transforms.Compose([transforms.ToTensor(), # Convert image to tensor
                                        transforms.Resize(new_size), # Resize
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize img
                                        ])

    # Initialize static datasets and dataloaders
    cats = Cats("all_cats", "test_iter_0.csv", "train_iter_0.csv")
    test_dataset = CatsDataset(root="all_cats",
                               imgs=cats.test_cats.index,
                               labels=cats.test_cats["label"].to_list(),
                               transform=img_transform)
    full_dataset = CatsDataset(root="all_cats",
                               imgs=cats.all_cats.index,
                               labels=None,
                               transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ConvNet(num_classes=len(cats.cls))
    teacher = TrainTest(model, device=device)

    # Class for results post processing
    pp = PostProcessor()

    # Overall
    mean_entropy = [] # Mean entropy of all predictions at each epoch
    total_loss_train = np.array([]) # loss of train data at each epoch
    total_loss_test = np.array([]) # loss of test data at each epoch
    total_score = np.array([]) # Scoring on test data at each epoch

    # Active learning training.
    iteration_counter = 0
    while True:
        print(f"==== ITERATION {iteration_counter} ====")
        # Prepare training dataset
        train_dataset = CatsDataset(root="all_cats",
                                    imgs=cats.train_cats.index,
                                    labels=cats.train_cats["label"].to_list(),
                                    transform=img_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training iteration on the prelabeled data
        loss_train, loss_test, score = teacher.learning_cycle(train_dataloader, test_dataloader, epochs=epochs)

        # Keep training summary
        total_loss_train = np.concatenate([total_loss_train, loss_train], axis=0)
        total_loss_test = np.concatenate([total_loss_test, loss_test], axis=0)
        total_score = np.concatenate([total_score, score], axis=0)

        # Making predictions on whole dataset
        print("Making predictions on whole dataset...")
        preds, preds_64 = teacher.predict(full_dataloader, num_classes=len(cats.cls))

        # Visualize 2d embeddings
        print("Preparing 2d embeddings and entropy plots")
        x_embedded_64, y_embedded_64, cmap = pp.vis_2d_tsne(preds, preds_64)
        fig_sum = plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.scatter(x_embedded_64, y_embedded_64, c=cmap, cmap="prism", edgecolors="black", linewidth=0.5)
        plt.title("TSNE: 64 dim-s embedding")

        # Mean predictions entropy
        mean_entropy.append(pp.mean_entropy(preds))

        # Plot the summary for active learning iteration
        plt.subplot(2, 2, 2)
        plt.scatter([*range(len(mean_entropy))], mean_entropy)
        plt.xlabel("Training iteration")
        plt.ylabel("Total predictions entropy")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot([*range(epochs*(iteration_counter + 1))], total_loss_train)
        plt.xlabel("Training iteration")
        plt.ylabel("Train loss")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot([*range(epochs * (iteration_counter + 1))], total_score)
        plt.xlabel("Training iteration")
        plt.ylabel("F1 score")
        plt.grid(True)
        print("Close image window to continue...")
        plt.show()

        # Save intermidiate results
        save_root = "Summary"
        cats.train_cats.to_csv(os.path.join(save_root, "train_data_iter_" + str(iteration_counter) + ".csv"))
        fig_sum.savefig(os.path.join(save_root, "embedding_entropy_iter_" + str(iteration_counter) + ".jpg"))

        # Stop or continue active learning
        stop = input(f"Would you like to continue training [y/n]?")
        if stop == "n":
            print("Done")
            break

        # Get and visualize the least reliable predictions
        print("Preparing worst predictions for manual labeling")
        worst_predictions = pp.worst_predictions(preds, n=n_worst)

        # Live plot for manual labeling
        print("Manual labeling...")
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        implot = ax.imshow(cats[0])

        for idx in range(len(worst_predictions)):
            fname = cats.all_cats.iloc[worst_predictions[idx]].name
            ax.set_title(fname)
            img = cats[worst_predictions[idx]]
            implot.set_data(img)
            implot.set_label(fname)
            fig.canvas.draw()
            fig.canvas.flush_events()

            label = input(f"[{fname}] \t kind=")
            cats.train_cats.loc[fname, "label"] = int(label)
        plt.ioff()
        plt.close()

        # Update active learning iteration counter
        iteration_counter += 1


if __name__ == "__main__":
    main()


