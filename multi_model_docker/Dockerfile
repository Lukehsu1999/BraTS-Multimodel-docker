# Base image with PyTorch + CUDA
FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL authors="Luke Hsu"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install nnUNet in editable mode
COPY nnUNet_install /app/nnUNet_install
WORKDIR /app/nnUNet_install
RUN pip install -e .

# Install dynamic-network-architectures in editable mode
COPY dynamic-network-architectures /app/dynamic-network-architectures
WORKDIR /app/dynamic-network-architectures
RUN pip install -e .

# Return to main app directory
WORKDIR /app

# Copy model checkpoint files and your scripts
COPY checkpoints /checkpoints
COPY main.py .
COPY infer_low_disk.py .

# Optional: sanity check to confirm files are present
RUN ls -la /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres

# (Quick and dirty) Fake path to satisfy hardcoded legacy model paths
RUN mkdir -p /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/dataset.json \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/ && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/plans.json \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/ && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/

# (Quick and dirty) Fake path for ResEncL model
RUN mkdir -p /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/dataset.json \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/ && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/plans.json \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/ && \
    cp /checkpoints/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth \
       /media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results/Dataset601_BraTS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/

# Set nnUNet environment variable
ENV nnUNet_results=/app/checkpoints

# Default runtime command — can override with your own at docker run
CMD ["python", "main.py", "--data_path", "/input", "--output_path", "/output", "--nnUNet_results", "/app/checkpoints"]
