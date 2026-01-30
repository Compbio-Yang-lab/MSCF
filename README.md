# MSCF

#——————Installation (Conda + pip – Recommended)——————
conda create -n deepliif_env python=3.8
conda activate deepliif_env
conda install -c conda-forge openjdk
pip install deepliif

#——————Install a PyTorch version compatible with your CUDA setup————————
import torch
torch.cuda.is_available()


#  Dataset Download
#  The official training and testing datasets can be downloaded from Zenodo:https://zenodo.org/record/4751737
#  Training data must follow the structure below
"""
Dataset/
├── train/
└── val/
"""

#——————Note on BC Dataset Splitting——————
For the BCDataset (breast cancer Ki67 dataset), MSCF does not use a dedicated test/ directory
Instead:

train/ → used for model training

val/ → used as the test set for evaluation

This design choice follows the original DeepLIIF implementation and is consistent with the released preprocessing scripts and experimental setup.


#——————————————Model Training Workflow——————————
deepliif train \
  --dataroot /path/to/Dataset \
  --name Model_Name

#——————————————Training Visualization and Outputs——————————————————
visdom -force_new_cookie
python train.py --dataroot /path/to/Dataset --name xxx  --seghead xxx  --display-server http://localhost --display-id 1 --gpu-ids 0 --display-env main




#——————————Test————————————
python test.py \
  --dataroot /path/to/test_images \
  --results_dir /path/to/results \
  --checkpoints_dir /path/to/model \
  --name Model_Name
