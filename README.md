# CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting

## TL;DR

CPiRi enables channel-permutation-invariant MTSF by combining frozen temporal encoding with lightweight spatial attention trained via channel shuffling, achieving SOTA accuracy with zero performance drop under dynamic sensor changes.

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=tgnXCCjKE3)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b)]() -->

## 📄 Paper Abstract
> *Current methods for multivariate time series forecasting can be classified into channel-dependent and channel-independent models. Channel-dependent models learn cross-channel features but often overfit the channel ordering, which hampers adaptation when channels are added or reordered. Channel-independent models treat each channel in isolation to increase flexibility, yet this neglects inter-channel dependencies and limits performance. To address these limitations, we propose CPiRi, a channel permutation invariant (CPI) framework that infers cross-channel structure from data rather than memorizing a fixed ordering, enabling deployment in settings with structural and distributional co-drift without retraining. CPiRi couples spatio-temporal decoupling architecture with permutation-invariant regularization training strategy: a frozen pretrained temporal encoder extracts high-quality temporal features, a lightweight spatial module learns content-driven inter-channel relations, while a channel shuffling strategy enforces CPI during training. We further ground CPiRi in theory by analyzing permutation equivariance in multivariate time series forecasting. Experiments on multiple benchmarks show state-of-the-art results. CPiRi remains stable when channel orders are shuffled and exhibits strong inductive generalization to unseen channels even when trained on only half of the channels, while maintaining practical efficiency on large-scale datasets. The source code and models will be released.*

## 🚀 Quick Start
Our experiments are built on [benchmark](https://github.com/GestaltCogTeam/BasicTS) $\text{BasicTS}^{+}$, please refer to [**Getting Started**](./tutorial/getting_started.md) for detailed environment setup and data preparation. $\text{BasicTS}^{+}$ (**Basic** **T**ime **S**eries) is a benchmark library and toolkit designed for time series forecasting. Our core codes are in [./baselines/CPiRi](./baselines/CPiRi) forder.

### 1. Environment Setup
```bash
conda create -n cpiri python=3.9
conda activate cpiri
pip install torch torchvision torchaudio transformers
pip install easy-torch easydict packaging setproctitle pandas scikit-learn tables sympy openpyxl setuptools numpy tqdm tensorboard transformers einops nvitop umap-learn
```

### 2. Data Preparation
You can download the `all_data.zip` file from [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1shA2scuMdZHlx6pj35Dl7A?pwd=s2xe). Unzip the files to the `datasets/` directory. These datasets have been preprocessed and are ready for use.

> The `data.dat` file is an array in `numpy.memmap` format that stores the raw time series data with a shape of [L, N, C], where L is the number of time steps, N is the number of time series, and C is the number of features.

> The `desc.json` file is a dictionary that stores the dataset’s metadata, including the dataset name, domain, frequency, feature descriptions, regular settings, and null values.

> Other files are optional and may contain additional information, such as `adj_mx.pkl`, which represents a predefined prior graph between the time series.

> If you are interested in the preprocessing steps, you can refer to the [preprocessing script](../scripts/data_preparation) and `raw_data.zip`.


For evaluating [Sundial](https://huggingface.co/thuml/sundial-base-128m/blob/main/model.safetensors) and [Timer-XL](https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/), please **download their offical pre-trained weights** to `baselines/Sundial/ckpt/model.safetensors` and `baselines/TimerXL/checkpoint.pth`. Training CPiRi needs `baselines/Sundial/ckpt/model.safetensors`.

### 3. Training
```bash
# Train on METR-LA
python experiments/train.py \
    -c baselines\CPiRi\METR-LA_LTSF.py \
    -gpu 0
```

### 4. File tree
```
├─baselines
│  ├─ChronosBolt
│  ├─CPiRi (Our core codes)
│  ├─Crossformer
│  ├─CrossGNN
│  ├─DLinear
│  ├─Informer
│  ├─iTransformer
│  ├─PatchTST
│  ├─STID
│  ├─Sundial (Baseline)
│  ├─TimerXL
│  └─TimeXer
├─basicts (The benchmark)
│  ├─data
│  ├─metrics
│  ├─runners
│  ├─scaler
│  └─utils
├─datasets (Download from benchmark)
├─experiments
│  ├─train.py (Run this!)
│  ├─evaluate.py
└─scripts
    └─data_preparation
       ├─CA
       ├─GBA
       ├─GLA
       ├─METR-LA
       ├─PEMS-BAY
       ├─PEMS04
       ├─PEMS08
       └─SD
```

## BibTeX
```bibtex
@inproceedings{iclr2026cpiri,
    title={CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting},
    author={Jiyuan Xu and Wenyu Zhang and Xin Jing and Jiahao Nie and Shuai Chen and Shuai Zhang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=tgnXCCjKE3},
    series = {ICLR '26}
}
```
