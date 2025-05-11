# MixSim3d:  A 3D Curriculum-Based Contrastive Learning Framework Applied to Digital Rock Physics

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/nguyenva04/mixsim3d_gretsi.svg)](https://github.com/nguyenva04/mixsim3d_gretsi/blob/main/LICENSE)

## 👥 Authors
- **Van Thao Nguyen**¹,² <van-thao.nguyen@ifpen.fr>
- **Dominique Fourer**² <dominique.fourer@univ-evry.fr>
- **Jean-François Lecomte**¹ <jean-francois.lecomte@ifpen.fr>
- **Souhail Youssef**¹ <souhail.youssef@ifpen.fr>
- **Désiré Sidibé**² <drodesire.sidibe@univ-evry.fr>

## Affiliations
¹ IFP Énergies nouvelles, Ruel Malmaison, France
² Laboratoire IBISC (EA 4526), Université d'Evry Paris-Saclay, Évry-Courcouronnes, France

## 📘 Overview

MixSim3d is an innovative deep learning self-supervised methodology designed to advance representation learning from 3D images, with a specific focus on Digital Rocks Physics (DRP) applications.

## 🚀 Features

- 🔬 Supports regression and classification tasks
- 💻 Distributed training
- 🧠 Flexible model configuration

## 📋 Prerequisites

-Refer to requirements.txt for installing all python dependencies. We use python 3.10.13 with pytorch 2.1.2.

## 🔧 Installation

```bash
git clone https://github.com/nguyenva04/mixsim3d_gretsi.git
cd mixsim3d_gretsi
pip install -r requirements.txt
pip install -e .
```
## 📂 Project Structure

```
mixsim3d_gretsi/
│
├── drp/                # Core package
│   ├── utils/
│   ├── builder/
│   ├── handlers/
│   ├── metrics/
│   └── train/
│
├── data/                
├── scripts/             # Execution scripts
└── README.md
```


# 🧪 Inference
This folder contains the dataset and pretrained model checkpoints used for inference on 3D images of sandstone rock (Boise), consisting of 100 subsamples of size 100×100×100.
This dataset is intended for use in inference tasks to predict permeability directly from the 3D volumetric data.

### 📁 Dataset Structure

Download here: [https://zenodo.org/records/15348378](https://zenodo.org/records/15348378)
```
DRP_gretsi.zip/
├── checkpoint/
│   ├── BYOL_checkpoint.pt
│   ├── MixSim3d_checkpoint.pt
│   ├── MoCo_checkpoint.pt
│   ├── ResNet18_checkpoint.pt
│   ├── SimCLR_checkpoint.pt
│   └── SimSiam_checkpoint.pt
├── minicubes/
│   ├── cube_000_offset_[1237_676_233].dat
│   ├── cube_001_offset_[0_52_783].dat
│   └── ...
└── minicubes_info.csv  # Contains: filename,x,y,z,label
```
### Full Example Command 
```bash
python ./scripts/run_inferent.py --config "path\mixsim3d_gretsi\drp\utils\cf\config_inferent.json"
```
📌 **Note** Modify the following in config_inferent.json:

"root_dir": Path to the small dataset

"checkpoint_path": Path to the pretrained model checkpoint

## 📊 Performance Metrics

- R2 Score (Regression)
- Top-K Accuracy (Classification)
- Scatter Plot Analysis
  
The results will be automatically saved in the MLflow logger when executing `train_finetune.py` and `predict.py`. 


## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Van Thao NGUYEN - [van-thao.nguyen@ifpen.fr](mailto:van-thao.nguyen@ifpen.fr)

Project Link: [https://github.com/nguyenva04/mixsim3d_gretsi](https://github.com/nguyenva04/mixsim3d_gretsi/tree/main)

For dataset access permissions, please contact us.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Distributed Training Frameworks](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)


