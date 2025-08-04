import os
import torch
from os import listdir
from os.path import join
from infer_low_disk import (
    convert_data_step, 
    perform_inference_step, 
    thresholding_step, 
    convert_back_BraTS_step
)

def infer():
    # === Standard input/output locations (BraTS requirement)
    data_path = "/input"
    output_path = "/output"

    # === Internal tmp folders
    tmp_path = join(output_path, "tmp")
    converted_dataset = join(tmp_path, "converted_dataset")
    inference_folder = join(tmp_path, "inference")

    os.makedirs(converted_dataset, exist_ok=True)
    os.makedirs(inference_folder, exist_ok=True)

    # === nnUNet model path
    os.environ['nnUNet_results'] = "/checkpoints"

    # === GPU Info
    print("GPU is available" if torch.cuda.is_available() else "GPU is not available")
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)

    # === Step 1: Convert raw to nnUNet format
    print("Step 1: convert_data_step")
    convert_data_step(input_folder_nnunet=converted_dataset, raw_dataset=data_path)
    print(f"Converted dataset files: {len(listdir(converted_dataset))}")

    # === Step 2: Inference
    print("Step 2: perform_inference_step")
    ensemble_code = 'ResNetM_ResNetL'  # update if needed
    perform_inference_step(
        inference_folder=inference_folder,
        input_folder_nnunet=converted_dataset,
        ensemble_code=ensemble_code
    )

    # === Step 3: Skipped in low-disk mode
    print("Step 3: skipping (ensemble accumulation handled already)")

    # === Step 4: Thresholding
    print("Step 4: thresholding_step")
    thresholding_step(
        min_volume_threshold_WT=250,
        min_volume_threshold_TC=150,
        min_volume_threshold_ET=100,
        inference_folder=inference_folder,
        ensemble_code=ensemble_code
    )

    # === Step 5: Convert final masks to BraTS format
    print("Step 5: convert_back_BraTS_step")
    convert_back_BraTS_step(
        min_volume_threshold_WT=250,
        min_volume_threshold_TC=150,
        min_volume_threshold_ET=100,
        inference_folder=inference_folder,
        ensemble_code=ensemble_code,
        brats_final_inference=output_path
    )

    print("DONE")

if __name__ == "__main__":
    infer()
