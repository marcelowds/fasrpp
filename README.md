# Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models

This project presents **FASR++**, a face super-resolution algorithm based on diffusion models that combines multiple low-resolution facial features through a learned aggregation network before the diffusion process. Compared with the original FASR, the proposed method replaces feature averaging with a neural feature aggregation strategy, leading to improved identity preservation and face recognition performance.

This project is a **fork and extension** of the original **FASR** implementation:

https://github.com/marcelowds/fasr

## Overview of the proposed method

<figure>
  <img src="https://raw.githubusercontent.com/marcelowds/fasrpp/main/fasrpp.pdf" style="width: 100%; max-width: 1000px;">
  <figcaption align="center">
    Fig. 1. Overview of the proposed FASR++ framework.
  </figcaption>
</figure>

## Qualitative Results

<figure>
  <img src="https://raw.githubusercontent.com/marcelowds/fasrpp/main/fasrpp_results.png" style="width: 100%; max-width: 1000px;">
  <figcaption align="center">
    Fig. 2. Comparison of low-resolution (LR), super-resolved (SR), and ground-truth (GT) images. FASR++ preserves facial identity and natural appearance while outperforming previous methods.
  </figcaption>
</figure>

<br><br>

This project extends the original FASR implementation, which was built upon forks of:

- [Score-SDE](https://github.com/yang-song/score_sde)
- [SDE-SR](https://github.com/marcelowds/sr-sde)

---

# Prepare the Conda environment

Create the environment:

```bash
conda create -n fasr python=3.8.2
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install JAX with CUDA support:

```bash
pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Activate the environment:

```bash
conda activate fasr
```

---

# Prepare TFRecords

The training images must be converted to TFRecords format. This can be done using the dataset generation tool from Progressive Growing of GANs:

```text
python dataset_tool.py create_from_images tfrecords_path images_path --shuffle 0
```

A small TFRecords example containing 10 CelebA images is available in:

```
sample_imgs/tfrecords
```

---

# AdaFace

Download the **IR-18 CASIA-WebFace** pretrained model from AdaFace:

https://github.com/mk-minchul/AdaFace

Place the downloaded checkpoint inside:

```
pretrained_adaface
```

---

# Pre-trained FASR++ model

Download the pretrained checkpoint and place it in:

```
exps/checkpoints-meta
```

---

# Sample images

The folder `sample_imgs` contains:

- `gallery` — gallery images used for face recognition;
- `LR_imgs` — low-resolution images used to extract identity features;
- `probe_HR` — ground-truth high-resolution probe images;
- `probe_LR` — low-resolution probe images used for super-resolution.

---

# Running the pipeline

Adjust the paths and parameters in:

```
configs/default_ve_configs.py
configs/ve/sr_ve.py
```

## 1. Create the TFRecords dataset

```text
python dataset_tool.py create_from_images tfrecords_path images_path --shuffle 0
```

## 2. Train the super-resolution model

```text
CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/ve/sr_ve.py --mode train --workdir exps
```

## 3. Generate super-resolved images

### 3.1 Extract AdaFace features

```text
python extract_features_LR.py
```

### 3.2 Combine the extracted features using the Delta network

```text
python combine_delta_network.py
```

### 3.3 Generate super-resolved images

```text
CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/ve/sr_ve.py --mode sr --workdir exps
```

---

# Citation

If you use this project, please cite:

> dos Santos, M., Laroca, R., Neves, J. C. R., & Menotti, D. Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models. *Journal of the Brazilian Computer Society*, 32(1), 1457–1470, 2026.

Journal version:
https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5884

arXiv:
https://arxiv.org/abs/2607.05702

```bibtex
@article{santos2025fasrpp,
  title={Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models},
  author={dos Santos, Marcelo and Laroca, Rayson and Neves, Jo{\~a}o Carlos Raposo and Menotti, David},
  journal={Journal of the Brazilian Computer Society},
  volume={32},
  number={1},
  pages={1457--1470},
  year={2026}
}
```
