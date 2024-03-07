# Cats Segmantation Problem
### Problem
There is a fully unlabeled dataset of cats images, it is necessary to visualize it in two-dimensional space so with separated clusters. Each cluster must correspond to a specific breed or color of the cat.
### Conditions
- The original dataset have to be splited to at least 6 clusters
- It is necessary to visualize the whole dataset
- The use of pre-trained models is not allowed
<p align="center">
  <img alt="img-name" src="https://github.com/OldFedot/CatsSegmentation/blob/master/Summary/cats_data_example.jpg" width="400">
  <br>
    <em>Fig. 1. Examples of cats images.</em>
</p>

### Summary
- The current repository provide the solution with using simple Convolutional NN and active learning approach.
- At first stage images were inspect manually to determine possible class candidates. Split to 9 different classes were chosen with respect to fur color.
    - Red 
    - White
    - Black
    - Grey
    - Grey-brown with stripes
    - White-red
    - White-black and white-grey
    - White-red-black
    - Siam
  
<p align="center">
  <img alt="img-name" src=https://github.com/OldFedot/CatsSegmentation/blob/master/Summary/cats_classes.jpg width="600">
  <br>
    <em>Fig. 2. Examples of cats images representing 9 different classes.</em>
</p>


## Repository structure:
- **[model.py](https://github.com/OldFedot/CatsSegmentation/blob/master/model.py)** is a file with model.
- **[train.py](https://github.com/OldFedot/CatsSegmentation/blob/master/train.py)** is a file with complete active learning training routine.
- **[analysis.py](https://github.com/OldFedot/CatsSegmentation/blob/master/analysis.py)** Performs evaluation of trained model.
- the notebook **[cats_segmentation_summary.ipynb](https://github.com/OldFedot/CatsSegmentation/blob/master/cats_segmentation_summary.ipynb)** Shows the training data, model and classification results.
- the csv **[train_iter_0.csv](https://github.com/OldFedot/CatsSegmentation/blob/master/train_iter_0.csv)** is a file with manual labeled data for the first iteration of active learning (~40 instances per class)
- the Summary **[train_iter_0.csv](https://github.com/OldFedot/CatsSegmentation/tree/master/Summary)** is a folder with graphs of model performance at each active learning iteration step, and csv files with corresponding manual labeld training data


### Dataset
I used the [SegmentedCats]([https://huggingface.co/datasets/huggan/wikiart](https://drive.google.com/file/d/1r7I9vculYHCd7x-FbnpQvPhmS2AvUSAI/view?usp=sharing)) dataset containing 1318 images of cats with removed background. All images have 256x256 resolution.

### Model
Simple convolutional net model We adapted 2D UNet model from Hugging Face [diffusers package](https://github.com/huggingface/diffusers) by adding three additional embedding layers to control paining style, including artist name, genre name and style name. Before adding the style embedding to time embedding, we pass each type of style embedding through [PreNet](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/5d37b3cc886aec9cfb077e4cb04cd3e7afaa536f/model.py#L14) modules. 

The network is trained to predict the unscaled noise component using Huber loss function (it produces better results on this dataset compared to L2 loss). During evaluation, the generated latent representations are decoded into images using the pretrained [VQ-VAE](https://arxiv.org/abs/1711.00937).
