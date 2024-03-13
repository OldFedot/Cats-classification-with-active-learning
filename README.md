# Cats Classification Problem
## Problem
There is a fully unlabeled dataset of cats images, it is necessary to visualize it in two-dimensional space so with separated clusters. Each cluster must correspond to a specific breed or color of the cat.

## Conditions
- The original dataset have to be splited to at least 6 clusters
- It is necessary to visualize the whole dataset
- The use of pre-trained models is not allowed

<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/assets/cats_data_example.jpg" width="400">
  <br>
    <em>Fig. 1. Examples of dataset instances.</em>
</p>

## Summary
- The current repository provide the solution with using simple Convolutional NN and active learning approach.
- At first, images were inspect manually to determine possible class candidates. Split to 9 different classes were chosen with respect to fur color: Red, White, Black, Grey, Grey-brown with stripes, White-red, White-black and white-grey, White-red-black, Siam (Figure 2.)

<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/assets/cats_classes.jpg" width="500">
  <br>
    <em>Fig. 2. Cats classes.</em>
</p>

- Around of 40 images of each class (360 in total) were selected and labeled manually for the first training iteration, while 10 images of each class (90 in total) were used for model evaluation.
- Further, at each active learning iteration, another 30 images corresponded to least reliable predictions were labeled and added to training data



### Active learning results
<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/assets/results_tsne_loss_entropy.gif" width="600">
  <br>
    <em>Fig. 3. Results of active learning at each iteration.</em>
</p>




### Repository structure:
- **[train.py](https://github.com/OldFedot/CatsSegmentation/blob/master/train.py)** is a file with complete active learning training routine.
- **[trainer.py](https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/trainer.py)** is a file with Trainer class which can perform a single iteration of active learning
- **[model.py](https://github.com/OldFedot/CatsSegmentation/blob/master/model.py)** is a file with model.
- **[analysis.py](https://github.com/OldFedot/CatsSegmentation/blob/master/analysis.py)** Performs evaluation of trained model.
- **[dataset.py](https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/dataset.py)** is file with classes responsible for data handling.
- the notebook **[cats_segmentation_summary.ipynb](https://github.com/OldFedot/CatsSegmentation/blob/master/cats_segmentation_summary.ipynb)** Shows the training data, model and classification results.
- the csv **[train_iter_0.csv](https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/data/seed/train_iter_0.csv)** is a file with manual labeled data for the first iteration of active learning (~40 instances per class)
- the Summary **[test_iter_0.csv](https://github.com/OldFedot/CatsClassificationWithActiveLearning/blob/master/data/seed/test_iter_0.csv)** is a folder with graphs of model performance at each active learning iteration step, and csv files with corresponding manual labeld training data


### Dataset
I used the **[SegmentedCats](https://drive.google.com/file/d/1r7I9vculYHCd7x-FbnpQvPhmS2AvUSAI/view?usp=sharing)** dataset containing 1318 images of cats with removed background. All images have 256x256 resolution.

