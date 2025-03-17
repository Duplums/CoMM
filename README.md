# What to align in multimodal contrastive learning ? 

[Benoit Dufumier*](https://scholar.google.fr/citations?user=r64E9-IAAAAJ) & [Javiera Castillo Navarro*](https://javi-castillo.github.io), [Devis Tuia](https://people.epfl.ch/devis.tuia), [Jean-Philippe Thiran](https://people.epfl.ch/jean-philippe.thiran)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.07402)
[![website](https://img.shields.io/badge/Website-Site-blue)](https://javi-castillo.github.io/comm)
[![Notebook Demo](https://img.shields.io/badge/Jupyter-Demo-orange?logo=jupyter)](https://github.com/Duplums/align_or_not/blob/main/demo/trifeatures.ipynb)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg)](#citation)

## Overview 

**Abstract**: Humans perceive the world through multisensory integration, blending the information of different modalities to adapt their behavior. Alignment through contrastive learning offers an appealing solution for multimodal self-supervised learning. Indeed, by considering each modality as a different view of the same entity, it learns to align features of different modalities in a shared representation space. However, this approach is intrinsically limited as it only learns shared or redundant information between modalities, while multimodal interactions can arise in other ways.
In this work, we introduce *CoMM*, a **Co**ntrastive **M**ulti**M**odal learning strategy that enables the **comm**unication between modalities in a single multimodal space. Instead of imposing cross- or intra- modality constraints, we propose to align multimodal representations by maximizing the mutual information between augmented versions of these multimodal features. Our theoretical analysis shows that shared, synergistic and unique terms of information naturally emerge from this formulation, allowing to estimate multimodal interactions beyond redundancy. We test CoMM both in a controlled and in a series of real-world settings: in the former, we demonstrate that CoMM effectively captures redundant, unique and synergistic information between modalities. In the latter, we show that CoMM learns complex multimodal interactions and achieves state-of-the-art results on seven multimodal tasks.

<div align="center">
  <span style="display: inline-block; background: white; padding: 10px; border-radius: 5px;">
  <img src="assets/CoMM_training.png" width="60%" />
  </span>
</div>

## Jupyter notebooks to reproduce our experiments 🚀

This repository contains different Jupyter notebooks to reproduce the experiments we performed in our paper. They are self-contained.

| Notebook | Description |
|----------|------------|
| [trifeatures.ipynb](demo/trifeatures.ipynb) | Controlled experiments on synthetic bimodal Trifeatures |
| [multibench.ipynb](demo/multibench.ipynb) | Real-world MultiBench experiments |
| [multibench_mimic.ipynb](demo/multibench_mimic.ipynb) | MIMIC experiments (require credentials) |
| [mmimdb.ipynb](demo/mmimdb.ipynb) | Text-Image MM-IMDb experiments  |
| [trimodal.ipynb](demo/trimodal.ipynb) | Vision&Touch experiments with 3 modalities |

---

## How to run the model locally ?

### Installation

You can install all the packages required to run CoMM with conda:

```sh
git clone https://github.com/Duplums/CoMM && cd CoMM
conda env create -f environment.yml
conda activate multimodal
```

### Controlled experiments on synthetic bimodal dataset

We first evaluate our proposed method CoMM against FactorCL, Cross and Cross+Self models on the synthetic bimodal Trifeatures dataset. It consists of pairs of images with controllable texture, shape and color. These pairs share shape as common features but with distinct colors and textures. Please check [this notebook](demo/trifeatures.ipynb) for more details.

#### Shell script

```sh
python3 main_trifeatures.py \
      +data=trifeatures \
      +data.data_module.biased=false \ # Set 'true' only for experiments on synergy
      +model=comm \
      model.model.encoder.embed_dim=512 \
      mode="train" \
      trainer.max_epochs=100
```

#### Results

<p float="left">
  <img src="assets/CoMM_results_trifeatures.png" width="50%" />
</p>


## Experiments on MultiBench

Then, we perform experiments on real-world MultiBench datasets including videos (audio, image, speech), tabular data (medical recordings), medical timeseries from ICU, force and proprioception data from robotics. Check [this notebook](demo/multibench.ipynb) for a demo.

#### Shell script

```sh
dataset="mosi" # Can be in ["mosi", "humor", "sarcasm", "mimic", "visionandtouch", "visionandtouch-bin"]
python3 main_multibench.py \
      data.data_module.dataset=${dataset} \
      model="comm" \
      trainer.max_epochs=100 \
      optim.lr=1e-3 \
      optim.weight_decay=1e-2
```

#### Results

Linear evaluation top-1 accuracy (in %, averaged over 5 runs) for classification tasks and MSE (×10⁻⁴) for regression task (V&T) on MultiBench after 100 epochs. 

---
| Model                 | V&T Reg↓  | MIMIC↑  | MOSI↑  | UR-FUNNY↑  | MUsTARD↑  | Average↑  |
|:--------------------:|:---------------------:|:--------:|:-----:|:----------:|:---------:|:-------------:|
| Cross                | 33.09                 | 66.7    | 47.8   | 50.1       | 53.5      | 54.52         |
| Cross+Self           | 7.56                  | 65.49   | 49.0   | 59.9       | 53.9      | 57.07         |
| FactorCL             | 10.82                 | **67.3** | 51.2   | 60.5       | 55.80     | 58.7         |
| **CoMM (ours)**      | **4.55**              | 66.4    | **67.5** | **63.1**  | **63.9**  | **65.22**    |
|                      |                       |         |        |            |           |               |
| 🟣SupCon        | -                     | 67.4    | 47.2   | 50.1       | 52.7      | 54.35          |
| 🟣FactorCL-SUP  | 1.72                  | **76.8** | 69.1   | 63.5       | 69.9      | 69.82         |
| 🟣**CoMM** | **1.34**          | 68.18   | **74.98** | **65.96** | **70.42** | **69.88**     |
---

Rows in 🟣 means supervised fine-tuning. Average is taken over classification results only.


## Experiments on MM-IMDb

Next, we focus on MM-IMDb, a large multimodal dataset for movie genre prediction, including image (movie poster) and text (plot) pairs. Each movie can be classified into one or more genres, thus the downstream task is multi-class multi-label with 23 categories. You can check [this notebook](demo/mmimdb.ipynb) for a demo. 

#### Shell script

```sh
python3 main_mmimdb.py \
      +model=comm \
      model.model.encoder.embed_dim=768 \
      +data=mmimdb \
      mode="train" \
      +mmimdb.encoders.1.mask_prob=0.15 \
      trainer.max_epochs=100 
```

#### Results

Linear evaluation for multi-class multi-label classification prediction of movie genres (in %, averaged over 5 runs) after 70 epochs.

---
| Model                                    | Modalities | Weighted-F1↑  | Macro-F1↑  |
|:----------------------------------------:|:----------:|:-------------:|:-----------:|
| SimCLR                                   | V          | 40.35         | 27.99       |
| CLIP                                     | V          | 51.5          | 40.8        |
| CLIP                                     | L          | 51.0          | 43.0        |
| CLIP                                     | V+L        | 58.9          | 50.9        |
| BLIP-2                                   | V+L        | 57.4          | 49.9        |
| SLIP                                     | V+L        | 56.54         | 47.35       |
| **CoMM (w/CLIP)**           | V+L        | _61.48_       | _54.63_     |
| **CoMM (w/BLIP-2)**         | V+L        | **64.75**     | **58.44**   |
|                                          |            |               |             |
| 🟣 **MFAS**                              | V+L        | 62.50         | 55.6        |
| 🟣 **ReFNet**                            | V+L        | -             | 56.7        |
| 🟣 **CoMM (w/CLIP)**  | V+L              | _64.90_       | _58.97_     |
| 🟣 **CoMM (w/BLIP-2)**| V+L              | **67.39**     | **62.0**    |
|                                          |            |               |             |
| LLaVA-NeXT                               | V+L        | 64.28         | 56.51       |
---
Rows in 🟣 means supervised fine-tuning.


## Experiments with 3 modalities

Finally, we perform CoMM on multimodal datasets with more than 3 modalities. We focus here on Vision&Touch, a robotic dataset with visual, force-torque and proprioception modalities when performing a peg insertion task. 150 trajectories are recorded, each of them consisting in 1000 timesteps. Check [this notebook](demo/trimodal.ipynb) for a demo.

#### Shell script
```sh
python3 main_multibench_all-mod.py \
      +model=comm \
      model.model.encoder.embed_dim=512 \
      +data=multibench \
      data.data_module.dataset="visionandtouch-bin" \
      mode="train" \
      trainer.max_epochs=100 
```

#### Results

Linear evaluation top-1 accuracy (%) on Vision&Touch and UR-FUNNY.

| Model                        | #Mod. | V&T CP | UR-FUNNY |
|------------------------------|-------|--------|----------|
| Cross                        | 2     | 84.4   | 50.1     |
| Cross+Self                   | 2     | 86.8   | 59.9     |
| CoMM *(ours)*                | 2     | *88.1* | *63.1*   |
| CMC                          | 3     | 94.1   | 59.2     |
| CoMM *(ours)*                | 3     | **94.2** | **64.6** |


## Citation

If you use CoMM, you may cite:

    @inproceedings{dufumier_castillo2025, 
        title={What to align in multimodal contrastive learning?},
        author={Dufumier, Benoit and Castillo-Navarro, Javiera and Tuia, Devis and Thiran, Jean-Philippe},
        booktitle={International Conference on Learning Representations},
        year={2025}
    }