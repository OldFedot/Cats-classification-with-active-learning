from sklearn.manifold import TSNE
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class PostProcessor():
    """
    Class for analysis of ConvNet model predictions: building TSNE, extracting worst predictions and mean predictions
    entropy.
    """
    def __init__(self):
        """Initialize TSNE."""
        self.tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)

    def vis_2d_tsne(self, preds, preds_64):
        """ Apply TSNE on 64 dimensions vector, and predictions.
        Parameters:
            preds (np.array): classifier predictions without Softmax layer.
            preds_64 (np.array): 64 dimensions representation of images.
        Returns:
            x_embedded_64 (np.array): x coordinates of embedded images.
            y_embedded_64 (np.array): y coordinates of embedded images.
            cmap (np.array): class predictions, used as colormap for further plotting.
        """
        cmap = np.zeros(shape=preds.shape[0])
        x_upd = np.zeros(shape=(preds.shape[0], preds.shape[1]))
        x_upd_64 = np.zeros(shape=(preds_64.shape[0], preds_64.shape[1]))

        for i in range(len(preds)):
            cmap[i] = np.argmax(sp.special.softmax(preds[i]))
            x_upd[i] = preds[i] / np.linalg.norm(preds[i])
            x_upd_64[i] = preds_64[i] / np.linalg.norm(preds_64[i])

        xy_embedded_64 = self.tsne.fit_transform(x_upd_64)
        x_embedded_64 = xy_embedded_64[:, 0]
        y_embedded_64 = xy_embedded_64[:, 1]
        return x_embedded_64, y_embedded_64, cmap

    def worst_predictions(self, preds, n=49):
        """
        Get the least reliable predictions.
        Parameters:
            preds (np.array): ConvNet model predictions without SoftMax.
            n (int): Number of N worst prediction to extract.
        Returns:
        """
        entropy = np.zeros(shape=len(preds))
        for i, pred in enumerate(preds):
            entropy[i] = self.get_entropy(sp.special.softmax(pred))
        high_N = (-entropy).argsort()[:n]
        return high_N

    def mean_entropy(self, preds):
        """
        Calculate mean entropy of predictions.
        Parameters:
             preds (np.array): Predictions of ConvNet model, with no SoftMax.
        Returns:
            entropy (float): Mean entropy of predictions
        """
        entropy = np.zeros(shape=len(preds))
        for i, pred in enumerate(preds):
            entropy[i] = self.get_entropy(sp.special.softmax(pred))
        return entropy.mean()

    def get_entropy(self, pred):
        """Calculate entropy of a single prediction."""
        logp = np.log2(pred)
        entropy = np.sum(-pred * logp)
        return entropy