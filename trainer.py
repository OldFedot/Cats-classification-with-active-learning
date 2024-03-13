from torchvision.transforms import transforms
from sklearn.metrics import f1_score
import numpy as np
import torch


class Trainer():
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