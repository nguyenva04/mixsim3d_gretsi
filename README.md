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

-Refer to requirements.txt for installing all python dependencies. We use python 3.10.13 with pytorch 2.1.2+cu118.

## 🔧 Installation

```bash
git clone https://github.com/nguyenva04/mixsim3d_gretsi.git
cd mixsim3d_gretsi
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

## 🏃 Self-supervised Training
To run the self-supervised pre-training for MixSim3d on 3D micro-CT datasets, use the `train.py` script with the following command structure:

```bash
python ./scripts/train_mixsim.py <dataset_path> -b <batch_size> -c <config_path> -e <epochs> -lr <learning_rate>
```

### Command Arguments

| Argument | Description | Example Value |
|----------|-------------|---------------|
| `<dataset_path>` | Path to the 3D micro-CT dataset | `"C:\Users\nguyenva\Documents\drp3d_project\data"` |
| `-b` | Batch size | `2` |
| `-c` | Path to configuration file | `"C:\Users\nguyenva\Documents\drp3d_project\drp\utils\cf\config_mixsim.json"` |
| `-e` | Number of training epochs | `100` |
| `-lr` | Learning rate | `1e-5` |

### Full Example Command
```bash
python ./scripts/train_mixsim.py "C:\Users\nguyenva\Documents\drp3d_project\data" -b 2 -c "C:\Users\nguyenva\Documents\drp3d_project\drp\utils\cf\config_mixsim.json" -e 100 -lr 1e-5
```
## 🎯 Fine-tuning  
To fine-tune the MixSim3d model on a 3D micro-CT dataset, use the `train_finetune.py` script with the following command structure:  

```bash
python ./scripts/train_finetune.py <dataset_path> -b <batch_size> -c <config_path> -e <epochs> -lr <learning_rate>
```

### Command Arguments  
| Argument | Description | Example Value |
|----------|-------------|---------------|
| `<dataset_path>` | Path to the 3D micro-CT dataset | `"C:\Users\nguyenva\Documents\drp3d_project\data"` |
| `-b` | Batch size | `2` |
| `-c` | Path to configuration file | `"C:\Users\nguyenva\Documents\drp3d_project\drp\utils\cf\config_finetune.json"` |
| `-e` | Number of training epochs | `100` |
| `-lr` | Learning rate | `1e-5` |

### Full Example Command  
```bash
python ./scripts/train_finetune.py "C:\Users\nguyenva\Documents\drp3d_project\data" -b 2 -c "C:\Users\nguyenva\Documents\drp3d_project\drp\utils\cf\config_finetune.json" -e 100 -lr 1e-5**
```

## 🔍 Prediction  
This step is to evaluate the model by producing the scatter plot, presenting the results in the paper.

```bash
python scripts/predict.py --config configs/config_predict.json 
```
**Note**: Please review and modify the checkpoint path in the `.json` file for fine-tuning and prediction steps.

---


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


