# Cats Segmantation Problem
### Problem
There is a fully unlabeled dataset of cats images, it is necessary to visualize it in two-dimensional space so with separated clusters. Each cluster must correspond to a specific breed or color of the cat.
### Conditions

- The original dataset have to be splited to at least 6 clusters
- It is necessary to visualize the whole dataset
- The use of pre-trained models is not allowed

<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsSegmentation/blob/master/Summary/cats_data_example.jpg" width="400" height="400>
  <br>
    <em>Fig. 1. Examples of cats images.</em>
</p>

## Summary

<p align="center">
  <img alt="img-name" src=https://github.com/OldFedot/CatsSegmentation/blob/master/Summary/cats_classes.jpg width="600">
  <br>
    <em>Fig. 2. Examples of cats images representing 9 different classes.</em>
</p>

- The current repository provide the solution with using simple Convolutional NN and active learning approach.
- At first, images were inspect manually to determine possible class candidates. Split to 9 different classes were chosen with respect to fur color.
    - Red 
    - White
    - Black
    - Grey
    - Grey-brown with stripes
    - White-red
    - White-black and white-grey
    - White-red-black
    - Siam
- Around of 40 images of each class (360 in total) were selected and labeled manually for the first itereation of training, and 10 images of each class (90 in total) was used for model evaluation.
- Further, at each active learning iteration, another 30 images corresponded to least reliable predictions were labeled and added to training data

### Active learning results
<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsSegmentation/blob/master/Summary/results_tsne_loss_entropy.gif" width="600">
  <br>
    <em>Fig. 3. Results of active learning at each iteration. TSNE embedding, mean entropy, Train loss and F1 score</em>
  </p>




## Repository structure:
- **[model.py](https://github.com/OldFedot/CatsSegmentation/blob/master/model.py)** is a file with model.
- **[train.py](https://github.com/OldFedot/CatsSegmentation/blob/master/train.py)** is a file with complete active learning training routine.
- **[analysis.py](https://github.com/OldFedot/CatsSegmentation/blob/master/analysis.py)** Performs evaluation of trained model.
- the notebook **[cats_segmentation_summary.ipynb](https://github.com/OldFedot/CatsSegmentation/blob/master/cats_segmentation_summary.ipynb)** Shows the training data, model and classification results.
- the csv **[train_iter_0.csv](https://github.com/OldFedot/CatsSegmentation/blob/master/train_iter_0.csv)** is a file with manual labeled data for the first iteration of active learning (~40 instances per class)
- the Summary **[train_iter_0.csv](https://github.com/OldFedot/CatsSegmentation/tree/master/Summary)** is a folder with graphs of model performance at each active learning iteration step, and csv files with corresponding manual labeld training data


### Dataset
I used the **[SegmentedCats](https://drive.google.com/file/d/1r7I9vculYHCd7x-FbnpQvPhmS2AvUSAI/view?usp=sharing)** dataset containing 1318 images of cats with removed background. All images have 256x256 resolution.

