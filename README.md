# RL-based-semantic-coding

This repository provides the official implementation of the paper:

**"Reinforcement Learning-Based Layered Lossy Image Semantic Coding"**


## Test with pre-trained model
To test the model, follow these steps:

#### 1. Download RL Model Weights

Download the RL model weights and place them in the `checkpoints` folder.

[RL model](https://drive.google.com/uc?export=download&id=1vjv4-J-PEEjoriWibgcLZ1rHIzq8Nlke)

#### 2. Download PSPNet Model Weights

Download the pre-trained PSPNet weights and place them in the `exp/cityscapes/pspnet101/model/` directory.

[PSPNet model](https://drive.google.com/uc?id=1FdQ_keCR1SjXtm1Co_BYV1wW3mf4fNx4)

#### 3. Run the Test Script
```bash
python test.py
```


## Train

#### 1. Prepare the Dataset

Download the **Cityscapes** dataset and place it in the `datasets/cityscapes` directory.

#### 2. Generate Images

Follow the instructions in the `semantic_image_synthesis/README.md` to generate reconstructed images from semantic maps. 

#### 3. Run the Training Script

```bash
python train.py 
```




