# BraTS 2025 Multi-Model Docker
## 🔗 References

- **Official BraTS Docker Submission Guide:**  
  https://www.synapse.org/Synapse:syn64153130/wiki/633742
- **Codebase Reference (GoAT Inference):**  
  https://github.com/ShadowTwin41/BraTS_2023_2024_solutions/tree/main/Segmentation_Tasks/BraTS_ISBI_GoAT_2024_inference

---
---

## 🛠️ Build and Run Docker Image

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
CUDA_VISIBLE_DEVICES="" docker run --rm --network none \
  --memory=16G --shm-size=4G \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/data:/input:ro" \
  -v "/mnt/c/Users/C0005/Desktop/brats25_segmentation_samples/7models_rap:/output:rw" \
  docker.synapse.org/syn68241107/nnunet-7models-rap:v1
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


