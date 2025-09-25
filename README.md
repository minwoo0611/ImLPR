<div align="center">

<h1>ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Model</h1>

📌 **[CoRL 2025]** This repository is the official repository for **ImLPR**  

📄 **[[Paper]](https://arxiv.org/abs/2505.18364)** | 🎥 **[[Video]](https://youtu.be/8t-hVO4yPdg)**  

<a href="https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko" target="_blank">Minwoo Jung</a>,  <a href="https://scholar.google.com/citations?user=fqfPCUkAAAAJ&hl=ko" target="_blank">Lanke Frank Tarimo Fu</a>,  <a href="https://scholar.google.com/citations?user=BqV8LaoAAAAJ&hl=ko" target="_blank">Maurice Fallon</a>,  <a href="https://scholar.google.co.kr/citations?user=7yveufgAAAAJ&hl=ko" target="_blank">Ayoung Kim</a><sup>†</sup>  

🤝 Collaboration with  **[Robust Perception and Mobile Robotics Lab (RPM)](https://rpm.snu.ac.kr/)** and **[Dynamic Robot Systems Group (DRS)](https://dynamic.robots.ox.ac.uk/)** 

</div>


### Recent Updates
- [2025/09/25] Code released!
- [2025/08/11] First release of ImLPR repository! 

### Contributions
Our work makes the following contributions:
1. **ImLPR is the first LPR pipeline using a VFM while retaining the majority of pre-trained knowledge**: Our key innovation lies in a tailored three-channel RIV representation and lightweight convolutional adapters, which seamlessly bridge the 3D LiDAR and 2D vision domain gap. Freezing most DINOv2 layers preserves pre-trained knowledge during training, ensuring strong generalization and outperforming task-specific LPR networks.
2. **We introduce the Patch-InfoNCE loss**: A patch-level contrastive loss to enhance the local discriminability and robustness of learned LiDAR features. We demonstrate that our patch-level contrastive learning strategy achieves a performance boost in LPR.
3. **ImLPR demonstrates versatility on multiple public datasets**: Outperforming SOTA methods. Furthermore, we also validate the importance of each component of the ImLPR pipeline, with code available post-review for robotics community integration.
   
<img width="2194" height="735" alt="Selection_1878" src="https://github.com/user-attachments/assets/b8172090-987c-4a2a-9c50-af164148ea69" />

---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Environment (Docker)](#environment-docker)
- [Data Preparation](#data-preparation)
  - [1) Folder layout](#1-folder-layout)
  - [2) Generate RIV images](#2-generate-riv-images)
  - [3) Build training/eval pickles](#3-build-trainingeval-pickles)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Weights](#pretrained-weights)
- [Results \& Notes](#results--notes)
- [Citations](#citations)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Environment (Docker)

Tested with **Ubuntu 22.04**, **CUDA 11.8**, **Python 3.11**, **PyTorch 2.6.0**, and GPUs such as RTX 3090 and A6000.

We provide a simple Dockerfile:

```bash
cd docker
sudo docker build -t imlpr:latest .
sudo docker run --gpus all -it --rm --env="DISPLAY" -e DISPLAY=:1.0 --net=host --ipc=host --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /:/mydata imlpr:latest
```

Adjust mount points to your needs.

---

## Data Preparation

### 1) Folder layout

Place your `.npy` RIV files and `poses.txt` in the following layout:

```
ImLPR/
└── data
    ├── training
    │   ├── SequenceA
    │   │   ├── 000000.npy
    │   │   └── poses.txt
    │   ├── SequenceB
    │   └── SequenceC
    ├── validation
    │   ├── SequenceA
    │   │   ├── 000000.npy
    │   │   └── poses.txt
    │   ├── SequenceB
    │   └── SequenceC
    ├── training.pickle
    ├── db_eval.pickle
    └── query_eval.pickle
```

**`poses.txt` format** (one line per frame):

```
timestamp x y z qx qy qz qw
```
**Note:** `poses.txt` is generated in **Step 2) Generate RIV & Trajectories** (see the preprocessing section), where timestamps and global poses are extracted from the dataset.

Each `.npy` is an `H × W × 3` RIV image. The three channels are **reflectivity (intensity)**, **range**, and **normal ratio**, each scaled to `[0, 255]` as described in the paper.

You can download the training/test sets and the corresponding pickle files from this **[Google Drive folder](https://drive.google.com/drive/folders/1dE-4qhZdMDHCq4wtR58htGfjVeUcxsqe?usp=sharing)** and then jump to the [Training](#training) section.

We provide:

* **Training sets (HeLiPR):** DCC, KAIST, Riverside
* **Evaluation sets (HeLiPR):** Roundabout01–03, Town01–03

---

### 2) Generate RIV images

If your raw LiDAR is not yet converted to RIV `.npy`, use your own converter or the helper in `preprocess/` (adapt paths inside as needed):

```bash
python3 preprocess/generate_riv_images.py
```

---

### 3) Build training/eval pickles

We provide two utilities:

- **Training tuples** → `data/training.pickle`
```bash
python3 preprocess/generate_training_tuples_baseline_img.py
```

- **Evaluation sets** → `data/db_eval.pickle` & `data/query_eval.pickle`
```bash
python3 preprocess/generate_test_sets_img.py
```

These scripts scan `data/training/*/poses.txt` and `data/validation/*/poses.txt`, to find the true positives for each sequences, and create pickles compatible with the training & eval code.

If you already have your own pickles, just place them at:
```
data/training_helipr.pickle
data/db_{dataset}_{sequence}.pickle
data/query_{dataset}_{sequence}.pickle
```

---

## Training

Use the provided configs:

- `config/config_helipr.txt` (training/eval params)  
- `config/config_model_imlpr.txt` (backbone/aggregator params)

Run:
```bash
cd training
python3 train.py \
  --config config/config_helipr.txt \
  --model_config config/config_model_imlpr.txt
```

Notes:
- Multi-GPU is supported via `torch.nn.DataParallel`.
- By default, the trainer will try to load `weights/ImLPR_default.pth` if present (remove/rename to start from scratch).

---

## Evaluation

Once trained (or using the default weights), evaluate:

```bash
cd eval
python3 evaluate.py \
  --config config/config_helipr.txt \
  --model_config config/config_model_imlpr.txt
```

This will compute average **Recall@1** and **One-Percent Recall** across configured sets.
Currently, **MulRan** and **HeLiPR** are supported out of the box.
Please adjust the dataset configuration in `eval/evaluate.py`.

---

## Pretrained Weights

Place pretrained weights (if provided) under:
```
weights/ImLPR_default.pth
```
Training and Evaluation will automatically load them (unless you change the loading logic in `training/trainer.py`). This checkpoint can be downloaded from the **[Google Drive folder](https://drive.google.com/drive/folders/1dE-4qhZdMDHCq4wtR58htGfjVeUcxsqe?usp=sharing)**.

---

## Results & Notes

- We freeze most **DINOv2** blocks and use lightweight **multi-conv adapters** to bridge LiDAR to vision.
- Training employs **Truncated Smooth-AP** for global retrieval and **Patch-InfoNCE** for local discriminability.
- Minor numerical differences can occur across GPUs/CUDA/flash-attn variants; typically within the first decimal place.

We will add more pretrained weights and evaluation files soon.

- [ ] Upload RIV files (.npy) for MulRan evaluation sequences
- [ ] Upload Python file to aggregate sparse LiDAR scans (used in zero-shot experiments)

---

## Citations

If you use **ImLPR**, please cite:

```bibtex
@INPROCEEDINGS { mwjung-2025-corl,
    AUTHOR = { Minwoo Jung and Lanke Frank Tarimo Fu and Maurice Fallon and Ayoung Kim },
    TITLE = { ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Models },
    BOOKTITLE = { Conference on Robot Learning (CoRL) },
    YEAR = { 2025 },
    MONTH = { Sep. },
    ADDRESS = { Seoul },
}
```

If you also use **HeLiPR** dataset, please cite:
```bibtex
@article{jung2024helipr,
  title={HeLiPR: Heterogeneous LiDAR dataset for inter-LiDAR place recognition under spatiotemporal variations},
  author={Jung, Minwoo and Yang, Wooseong and Lee, Dongjae and Gil, Hyeonjae and Kim, Giseop and Kim, Ayoung},
  journal={The International Journal of Robotics Research},
  volume={43},
  number={12},
  pages={1867--1883},
  year={2024},
  publisher={SAGE}
}
```

---

## Contact

Questions or issues?  
- Open a GitHub issue, or  
- Email: **moonshot@snu.ac.kr**

---

## Acknowledgments

Our codebase builds upon great open-source projects:
- **[MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2)**
- **[SelaVPR++ (Paper)](https://arxiv.org/abs/2502.16601)**
- **[SALAD](https://github.com/serizba/salad)**

Thanks to the communities for sharing code and insights.