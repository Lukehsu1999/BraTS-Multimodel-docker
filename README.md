# 🧠 BraTS GoAT Challenge – 1st Place Solution (MICCAI 2025)  🏆
**Enhancing Brain Tumor Segmentation Generalizability via Pseudo-Labeling and Ratio-Adaptive Postprocessing**  
*To-Liang Hsu¹⋆, Dang Khoa Nguyen¹⋆, Pai Lin¹, Ching-Ting Lin¹, Wei-Chun Wang¹⋆⋆*  
¹ China Medical University Hospital Artificial Intelligence Center, Taichung, Taiwan  
(⋆ Equal contribution, ⋆⋆ Corresponding author)  
*Accepted to MICCAI 2025 (in press, Lecture Notes in Computer Science, Springer)*  

📄 [**Preprint (coming soon)**]() &nbsp;|&nbsp; 🏆 [**BraTS 2025 Official Website**](https://www.synapse.org/Synapse:syn64153130/wiki/630130)

<p align="center">
  <img src="https://github.com/Lukehsu1999/Lukehsu1999/blob/main/MICCAI_Presentation.jpg" width="45%" valign="middle"/>
  <img src="https://github.com/Lukehsu1999/Lukehsu1999/blob/main/MICCAI_BraTS_Trophy.png" width="45%" valign="middle"/>
</p>


## 🧭 Repository Overview

The [**BraTS - GoAT Challenge**](https://www.synapse.org/Synapse:syn64153130/wiki/631456) (*Brain Tumor Segmentation - Generalizability Across Tumors*) is an international competition at **MICCAI — Medical Image Computing and Computer-Assisted Intervention**, evaluating how well brain tumor segmentation algorithms generalize across **tumor types, demographics, and institutions**.

Our approach introduces three key contributions that collectively enhance generalizability across heterogeneous tumor domains:

1. **Pseudo-Label Supervised Fine-Tuning** — leveraging unlabeled MRI cases through conservative pseudo-label generation to expand the effective training distribution while maintaining label reliability.  
2. **Ratio-Adaptive Postprocessing** — a **case-customized yet tumor-type-agnostic** refinement method that dynamically scales thresholds based on predicted tumor volume, **without requiring prior tumor-type knowledge**, achieving a balanced precision–recall tradeoff across heterogeneous cases.  
3. **TumorSurfer (Exploratory Module)** — a multitask, anatomy-aware framework that jointly learns tumor and structural priors, offering insights for future research on anatomy-guided segmentation.

This repository provides the **official Dockerized implementation** of our **1st-place solution**, containing the complete inference and evaluation pipeline submitted to the MICCAI evaluation server.
> ⚠️ **Note:** Model checkpoints are **not included** in this repository due to their large size. Please contact the authors if access is required.

Experimental training pipelines and ablation studies were developed in a separate internal repository by **Luke Hsu** and **Dang Khoa Nguyen**.  
For research inquiries or collaboration, please contact the authors.


## 📘 Table of Contents
- [Problem Statement](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#-problem-statement)
- [Methods](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#-methods)
  - [Pseudo-Label Supervised Finetuning](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#-pseudo-label-supervised-fine-tuning)
  - [Ratio-Adaptive Postprocessing](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#%EF%B8%8F-ratio-adaptive-postprocessing)
- [Exploratory Module: TumorSurfer](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/edit/main/README.md#-exploratory-module-tumorsurfer)
- [Results](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#-results)
- [References for Dockerization](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/edit/main/README.md#-references-for-dockerization)
- [Build and Run Docker Image](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#%EF%B8%8F-build-and-run-docker-image)
- [Common Setup & Debugging Issues](https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/README.md#-common-setup--debugging-issues)

## 🧩 Problem Statement
The **MICCAI BraTS GoAT Challenge** (*Brain Tumor Segmentation Generalizability Across Tumors*) evaluates how well segmentation models **generalize across tumor types**, a key step toward clinically reliable AI.

Unlike earlier BraTS editions limited to adult gliomas, GoAT includes:
**Adult gliomas**, **African gliomas**, **Meningiomas**, **Brain metastases**, and **Pediatric tumors** — each differing in scanner type, lesion pattern, and demographics.

The goal is to build algorithms that can:
- Segment key **tumor subregions** (*ET (Enhancing Tumor)*, *NETC (Non-Enhancing Tumor Core)*, *SNFH (Surrounding FLAIR Hyperintensity / Edema)*)  
- Perform robustly **across institutions and tumor types**
  
<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/Radiopaedia_Glioma.jpeg" width="95%"/><br/>
      <sub><b>(a)</b> Glioma </sub>
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/Radiopaedia_Meningioma2.jpg.png" width="95%"/><br/>
      <sub><b>(b)</b> Meningioma </sub>
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/Radiopaedia_Metastasis.jpeg" width="95%"/><br/>
      <sub><b>(c)</b> Metastasis </sub>
    </td>
  </tr>
</table>
<p align="center">
  <em> 
  Images adapted from <a href="https://radiopaedia.org/" target="_blank">Radiopaedia.org</a></em>
</p>

---

## 🧠 Methods

Our segmentation system improves **cross-tumor generalization** through two core modules — **Pseudo-Label Supervised Fine-Tuning** and **Ratio-Adaptive Postprocessing** — built atop a diverse ensemble of nnU-Net and U-Mamba architectures.  
These components complement each other: pseudo-labels expand supervision to unseen tumor patterns, while ratio-adaptive rules refine predictions without relying on tumor-type information.


### 🧩 Pseudo-Label Supervised Fine-Tuning
**Motivation:**  
The labeled BraTS-GoAT dataset captures only part of the tumor diversity across glioma, meningioma, metastasis, and pediatric cases.  
To better represent unseen tumor phenotypes, we leveraged unlabeled scans via pseudo-labeling.

**Pipeline:**  
1. Train three **nnU-Net ResEnc** models (M/L/XL) on labeled data.  
2. **Ensemble** their softmax outputs to stabilize predictions (with official validation lesion-wise Dice of 0.82).  
3. Apply **conservative filtering** — removing only components < 10 voxels to preserve sensitivity.  
4. Merge the refined pseudo-labels with the labeled set → a mixed dataset of 2,489 cases.  
5. Fine-tune ResEnc-L/XL and U-MambaBot models on this combined set.

**Insight:**  
Pseudo-label supervision improved generalization, especially for **Whole Tumor (WT)** segmentation, by exposing the model to greater lesion variability.  
While effects on fine subregions were modest, pseudo-labels enriched the ensemble’s diversity and robustness across domains.

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/PseudoLabelingPipeline.png" width="75%">
</p>
<p align="center"><em>Pseudo-label generation pipeline</em></p>


### ⚖️ Ratio-Adaptive Postprocessing
**Challenge:**  
Fixed thresholding rules from prior BraTS challenges fail under GoAT’s cross-tumor variability — a single cutoff cannot fit both large gliomas and small metastases.

**Approach:**  
We designed a **ratio-adaptive thresholding** strategy that scales cutoffs based on each case’s predicted tumor volume:

```
WT_vol = average predicted Whole Tumor component size in this case
ET_thresh = min(0.0005 * WT_vol, 100)
WT_thresh = max(min(0.005 * WT_vol, 250), 10)
```

This formulation blends literature-inspired bounds with adaptive scaling, removing small noisy components while preserving valid small lesions.

**Effect:**  
Compared with fixed thresholds, ratio-adaptive filtering reduced **ET false positives** significantly while limiting false negatives growth, yielding a more balanced precision–recall trade-off and higher lesion-wise Dice.  
The strategy proved stable across all tumor types without needing explicit type labels.

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/BraTSFinalSystem.jpg" width="80%">
</p>
<p align="center"><em>Overall System Overview</em></p>

---

## 🌊 Exploratory Module: TumorSurfer
**Concept:**  
To bridge tumor and healthy-tissue understanding, we explored **TumorSurfer**, a multitask model jointly predicting anatomical structures and tumor subregions.  
Using **FastSurfer-derived anatomical labels** refined by our own *SimpleSurfer* network, TumorSurfer encouraged structural awareness in boundary learning.

Although it achieved **Dice > 0.9** for major anatomical regions (white matter, cortex, ventricles), tumor segmentation lagged due to label imbalance and shared-capacity effects.  
Future directions include coupling anatomical priors with the tumor model through attention or anatomy-aware postprocessing.

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/TumorSurfer-FastSurfer.jpg" width="75%">
</p>
<p align="center"><em>Pre-trained FastSurfer for rough anatomical labels. Note that FastSurfer accepts T1 MRI.</em></p>

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/TumorSurfer-SimpleSurfer.jpg" width="75%">
</p>
<p align="center"><em> Train SimpleSurfer for refined anatomical labels, accepting 4 MRI modalities as input</em></p>

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/FastSurfer_SimpleSurfer.png" width="75%">
</p>
<p align="center"><em> Visual comparison between pre-trained FastSurfer and our customized SimpleSurfer</em></p>

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/TumorSurfer-TumorSurfer.jpg" width="75%">
</p>
<p align="center"><em> Train TumorSurfer to segment both anatomical and tumor labels</em></p>

<p align="center">
  <img src="https://github.com/Lukehsu1999/BraTS-Multimodel-docker/blob/main/diagrams/TumorSurfer.png" width="75%">
</p>
<p align="center"><em> Example of TumorSurfer output</em></p>

--- 
## 🧪 Results

### 🎯 Official Test-Phase Results (BraTS 2025 GoAT)
| Metric | ET | TC | WT |
|:--|:--:|:--:|:--:|
| Lesion-wise Dice | 0.800 | 0.827 | 0.828 |
| Lesion-wise NSD @ 1mm | 0.826 | 0.801 | 0.763 |

> 🏆 Co-first place at the MICCAI BraTS 2025 Lighthouse GoAT Challenge.  
> Evaluated on hidden test data by the BraTS organizers.


### 💡 Validation Results (Before vs. After Postprocessing)
| Model | ET | NETC | SNFH | TC | WT |
|:--|:--:|:--:|:--:|:--:|:--:|
| 7-model Ensemble (Raw) | 0.745 | 0.662 | 0.782 | 0.824 | 0.816 |
| 7-model Ensemble (Postprocessed) | **0.756** | 0.662 | **0.788** | **0.827** | **0.831** |

> Ratio-adaptive postprocessing improved lesion-wise Dice across all tumor subregions.


---

## 🔗 References for Dockerization

- **Official BraTS Docker Submission Guide:**  
  https://www.synapse.org/Synapse:syn64153130/wiki/633742
- **Codebase Reference (GoAT Inference):**  
  https://github.com/ShadowTwin41/BraTS_2023_2024_solutions/tree/main/Segmentation_Tasks/BraTS_ISBI_GoAT_2024_inference

---

## 🛠️ Build and Run Docker Image
### Pre-requests:
This repository is mainly for the official dockerization of the models, it assumes you already have the 7 models trained and checkpoints saved. <br>
Feel free to reach out to me for more details on model training.
```
        "ResNetM": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        "ResNetL": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        "ResNetXL": "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
        "FinetuneM": "nnUNetTrainerDiceTopK10FocalFineTuning1000Eps__nnUNetResEncUNetMPlans__3d_fullres",
        "FinetuneL": "nnUNetTrainerDiceTopK10FocalFineTuning__nnUNetResEncUNetLPlans__3d_fullres",
        "FinetuneXL": "nnUNetTrainerDiceTopK10FocalFineTuning__nnUNetResEncUNetXLPlans__3d_fullres",
        "UMamba": "nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres"
```
Open your Ubuntu terminal

Clean up (optional):

```
# Remove old image by tag
docker image rm nnunet-resnetm:v2

# Prune unused build cache
docker builder prune -f
```
Build the image:
```
cd multi_model_docker/
docker build -t nnunet-resnetm-resnetl:v5 .
```
Run the Docker:
According to BraTS
```
docker run --rm --network none --gpus all \
  --memory=16G --shm-size=4G \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output:rw" \
  nnunet-resnetm-resnetl:v5 
```
Run locally:
```
docker run --rm --network none --gpus all \
  --memory=16G --shm-size=4G \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/data:/input:ro" \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/7models_rap:/output:rw" \
  nnunet-7models-rap:v1
```
```
docker run \
    --rm \
    --network none \
    --runtime=nvidia \
    --env NVIDIA_VISIBLE_DEVICES=0 \
    -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/data:/input:ro" \
    -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/7models_rap:/output:rw" \
    --memory=16G --shm-size 4G \
    docker.synapse.org/syn68241107/nnunet-7models-rap:v1
```
```
docker run --rm --network none --gpus all \
  --memory=16G --shm-size=4G \
  -v "/mnt/c/Users/C0005/Desktop/subtype_input:/input:ro" \
  -v "/mnt/c/Users/C0005/Desktop/5models_rap:/output:rw" \
  nnunet-5models-rap:v1
```
Run the uploaded docker:
```
docker run --rm --network none --gpus all \
  --memory=16G --shm-size=4G \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/data:/input:ro" \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/7models_rap:/output:rw" \
  docker.synapse.org/syn68241107/nnunet-7models-rap:v1
```
Run the uploaded docker (without GPU):
```
nnunet-7models-rap-cpu:v1 python main.py --device cpu docker run --rm --network none --memory=16G --shm-size=4G -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/data:/input:ro" -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/7models_rap_cpu:/output:rw" nnunet-7models-rap-cpu:v1 python main.py --device cpu
```
With extra shared memory (16GB), mounted on input and output
```
docker run --rm --gpus all \
  --shm-size=16g \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  -v "$PWD/tmp:/app/tmp" \
  nnunet-resnetm-resnetl:v3 \
  python -u main.py --data_path /input --output_path /output --nnUNet_results /app/checkpoints

```

## 🧱 Common Setup & Debugging Issues

This section documents all major problems I encountered while containerizing a single nnUNet model for the BraTS GoAT challenge. These notes help ensure full reproducibility and traceability for future reference.

---

### 1. 🚫 Docker & GPU Incompatibility

- **Problem:** My JupyterHub environment had a GPU but did **not** allow Docker access.  
  My local Windows desktop supported Docker but had **no GPU**.
- **Resolution:**  
  Borrowed a laptop with an **NVIDIA RTX 4090 GPU** and installed Docker + NVIDIA Container Toolkit manually.

---

### 2. ⚙️ Ubuntu WSL2 + NVIDIA Container Toolkit Setup

- **Problem:**  
  Running Docker with GPU on Windows requires:
  - Ubuntu installed under WSL2
  - Installation of the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

### 3. 📦 Missing `dynamic-network-architectures` in `nnUNet`

- **Problem:**  
  When installing from a copied `nnUNet` repo, the following error appears:
```
ModuleNotFoundError: No module named 'dynamic_network_architectures'
```
- **Root Cause:**  
`dynamic-network-architectures` is a separate required repo.
- **Resolution:**
- Cloned [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) into the project folder
- During Docker containerization, I **copied both repos** (`nnUNet` and `dynamic-network-architectures`) into the image
- Used `pip install -e` to install them properly

---

### 4. 📁 nnUNet Model Path Reference Error

- **Problem:**  
Even after copying the trained model, `plans.json`, and `dataset.json` into my Docker repo, the nnUNet code **still references the original host path**, e.g.:
```
FileNotFoundError: [Errno 2] No such file or directory: '/media/volume1/.../dataset.json'
```
- **Dirty Fix (Temporary):**
- Inside the Docker container, I **recreated the original folder path**, e.g. `/media/volume1/.../`, and placed the `dataset.json` there.
- Copied `nnUNet` results into this fake directory for the model to run.
- **Note:**  
This issue is unresolved properly — ideally, I want to modify the nnUNet logic to accept arbitrary paths.

- ** Final Fix:**
fix the fixed path bug in nnUNet_install/nnUNetv2/paths.py
---

### 5. 💥 "Bus Error (Core Dump)" During Inference

- **Problem:**  
When running inference via Docker, the container crashes with:
```
Bus error (core dumped)
```
- **Root Cause:**  
Docker ran out of allocated memory (especially common in Ubuntu + WSL2 setup).
- **Resolution:**
- Increased Docker memory allocation by editing `.wslconfig` in the Windows user home directory:
  ```
  [wsl2]
  memory=16GB
  processors=6
  swap=8GB
  ```
- Restarted WSL:
  ```bash
  wsl --shutdown
  ```

---


