# DermaTransNet
DermaTransNet: Where Transformer Attention Meets U-Net for Skin Image Segmentation

## Official Implementation of DermaTransNet

### Details of Model
This study introduces a novel attention-based encoder-decoder architecture designed for precise segmentation of skin layers (Epidermis, Dermis, Hypodermis, Keratin) from stained whole slide image samples. The proposed Transformer-based encoder leverages a multi-axis structure to effectively capture both global and local features, which are then transmitted to the decoder through an attention-based gated skip connection. The attention-mixing decoder integrates multi-head self-attention, spatial attention, and squeeze excitation modules to enhance spatial information gain and refine segmentation accuracy.
     	
### Requirements
- loguru
- tqdm
- pyyaml
- pandas
- matplotlib
- scikit-learn
- scikit-image
- scipy
- opencv-python
- seaborn
- albumentations
- tabulate
- warmup-scheduler
- torch==1.11.0+cu113
- torchvision==0.12.0+cu113
- mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
- timm
- einops
- pthflops
- torchsummary
- thop

### Datasets
This study uses the following Datasets:
- QueensLand Dataset  https://espace.library.uq.edu.au/view/UQ:8be4bd0
- HistoSeg Dataset https://data.mendeley.com/datasets/vccj8mp2cg/1

### Preparing the data for training
Whole slide image samples were patched into 256x256 sized patches to reduce computational complexity.

### Training

To train the model, Run trainer.py


### Testing (Model Evaluation)

For testing the model, Run test.py

