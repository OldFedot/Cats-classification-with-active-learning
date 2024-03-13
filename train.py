import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import ConvNet
from analysis import Analyzer
from dataset import Cats, CatsDataset
from trainer import Trainer


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
    cats = Cats("data/all_cats",
                "data/seed/test_iter_0.csv",
                "data/seed/train_iter_0.csv")
    test_dataset = CatsDataset(root="data/all_cats",
                               imgs=cats.test_cats.index,
                               labels=cats.test_cats["label"].to_list(),
                               transform=img_transform)
    full_dataset = CatsDataset(root="data/all_cats",
                               imgs=cats.all_cats.index,
                               labels=None,
                               transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ConvNet(num_classes=len(cats.cls))
    teacher = Trainer(model, device=device)

    # Class for results post processing
    pp = Analyzer()

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
        train_dataset = CatsDataset(root="data/all_cats",
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
        print("Preparing 2d embeddings and entropy plots...")
        x_embedded_64, y_embedded_64, cmap = pp.vis_2d_tsne(preds, preds_64)
        fig_sum = plt.figure(figsize=(10, 10))
        fig_sum.suptitle(f"Iteration: [{iteration_counter}], "
                         f"Train size = [{(100 * len(train_dataloader.dataset) / len(full_dataloader.dataset)):.1f}%]")
        plt.subplot(2, 2, 1)
        plt.scatter(x_embedded_64, y_embedded_64, c=cmap, cmap="prism", edgecolors="black", linewidth=0.5)
        plt.title("TSNE: 64 dim-s embedding")

        # Mean predictions entropy
        mean_entropy.append(pp.get_mean_entropy(preds))

        # Plot the summary for active learning iteration
        plt.subplot(2, 2, 2)
        plt.scatter([*range(len(mean_entropy))], mean_entropy)
        plt.xlabel("Training iteration")
        plt.ylabel("Mean predictions entropy")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot([*range(epochs*(iteration_counter + 1))], total_loss_train)
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot([*range(epochs * (iteration_counter + 1))], total_score)
        plt.xlabel("Epoch")
        plt.ylabel("F1 score")
        plt.grid(True)
        print("Close image window to continue...")
        plt.show()

        # Save intermidiate results
        save_root = "data/output"
        cats.train_cats.to_csv(os.path.join(save_root, "train_data_iter_" + str(iteration_counter) + ".csv"))
        fig_sum.savefig(os.path.join(save_root, "embedding_entropy_iter_" + str(iteration_counter) + ".jpg"))

        # Stop or continue active learning
        stop = input(f"Would you like to continue training [y/n]?")
        if stop == "n":
            print("Done")
            break

        # Get and visualize the least reliable predictions
        print("Preparing worst predictions for manual labeling")
        worst_predictions = pp.get_worst_predictions(preds, n=n_worst)

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


