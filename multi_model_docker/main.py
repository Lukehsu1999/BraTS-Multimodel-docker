import os
import argparse
import torch
from os import listdir
from os.path import join
from infer_low_disk import (
    convert_data_step, 
    perform_inference_step, 
    thresholding_step, 
    convert_back_BraTS_step
)

def infer(data_path, output_path, converted_dataset_path, inference_folder_path):
    # Set the path for the weights
    os.environ['nnUNet_results'] = "/checkpoints"

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)

    # Step 1: Convert dataset from BraTS2023 to nnUNet format
    print("Doing first step: convert_data_step")
    convert_data_step(input_folder_nnunet=converted_dataset_path, raw_dataset=data_path)
    print(f"Converted dataset files: {len(listdir(converted_dataset_path))}")

    # Step 2: Inference for each model
    print("Doing second step: perform_inference_step")
    ensemble_code = 'ResNetM_ResNetL'
    perform_inference_step(
        inference_folder=inference_folder_path,
        input_folder_nnunet=converted_dataset_path,
        ensemble_code=ensemble_code
    )
    print("Inference complete.")

    # Step 3: Skipped in low-disk mode
    print("Skipping third step (already done in inference loop)")

    # Step 4: Thresholding
    print("Doing fourth step: thresholding_step")
    thresholding_step(
        min_volume_threshold_WT=250,
        min_volume_threshold_TC=150,
        min_volume_threshold_ET=100,
        inference_folder=inference_folder_path,
        ensemble_code=ensemble_code
    )

    # Step 5: Convert to final BraTS output
    print("Doing fifth step: convert_back_BraTS_step")
    convert_back_BraTS_step(
        min_volume_threshold_WT=250,
        min_volume_threshold_TC=150,
        min_volume_threshold_ET=100,
        inference_folder=inference_folder_path,
        ensemble_code=ensemble_code,
        brats_final_inference=output_path
    )
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS inference entrypoint")
    parser.add_argument("--data_path", type=str, help="Path to raw input data")
    parser.add_argument("--output_path", type=str, help="Path to save predictions")
    args = parser.parse_args()

    # Set up working paths
    tmp_path = join(args.output_path, "tmp")
    converted_dataset = join(tmp_path, "converted_dataset")
    inference_folder = join(tmp_path, "inference")

    os.makedirs(converted_dataset, exist_ok=True)
    os.makedirs(inference_folder, exist_ok=True)

    infer(
        data_path=args.data_path,
        output_path=args.output_path,
        converted_dataset_path=converted_dataset,
        inference_folder_path=inference_folder
    )
