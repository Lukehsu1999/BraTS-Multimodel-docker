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

#ENV nnUNet_results=/media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/nnUNet_results
ENV nnUNet_results=/checkpoints

# Default runtime command — can override with your own at docker run
CMD ["python", "-u", "main.py"]

