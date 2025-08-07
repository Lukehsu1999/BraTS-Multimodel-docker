import os
import torch
from os import listdir
from os.path import join
import shutil
from infer_low_disk import (
    convert_data_step, 
    perform_inference_step, 
    #thresholding_step, 
    #convert_back_BraTS_step,
    ratio_adaptive_postprocessing_step,
    convert_back_BraTS_step_ratio_adaptive
)
import argparse

def infer(device):
    # === Standard input/output locations (BraTS requirement)
    data_path = "/input"
    output_path = "/output"

    # === Internal tmp folders
    tmp_path = join(output_path, "tmp")
    converted_dataset = join(tmp_path, "converted_dataset")
    inference_folder = join(tmp_path, "inference")

    os.makedirs(converted_dataset, exist_ok=True)
    os.makedirs(inference_folder, exist_ok=True)


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
    ensemble_code = 'ResNetM_ResNetL_ResNetXL_FinetuneM_FinetuneL_FinetuneXL_UMamba'  # update if needed
    perform_inference_step(
        inference_folder=inference_folder,
        input_folder_nnunet=converted_dataset,
        ensemble_code=ensemble_code,
        device=device
    )

    # === Step 3: Skipped in low-disk mode
    print("Step 3: skipping (ensemble accumulation handled already)")

    # # === Step 4: Thresholding
    # print("Step 4: thresholding_step")
    # thresholding_step(
    #     min_volume_threshold_WT=250,
    #     min_volume_threshold_TC=150,
    #     min_volume_threshold_ET=100,
    #     inference_folder=inference_folder,
    #     ensemble_code=ensemble_code
    # )

    # # === Step 5: Convert final masks to BraTS format
    # print("Step 5: convert_back_BraTS_step")
    # convert_back_BraTS_step(
    #     min_volume_threshold_WT=250,
    #     min_volume_threshold_TC=150,
    #     min_volume_threshold_ET=100,
    #     inference_folder=inference_folder,
    #     ensemble_code=ensemble_code,
    #     brats_final_inference=output_path
    # )
    # === Step 4: Ratio-Adaptive Postprocessing
    print("Step 4: ratio_adaptive_postprocessing_step")
    ratio_adaptive_postprocessing_step(
        inference_folder=inference_folder,
        ensemble_code=ensemble_code,
        save_csv=True  # True for local test, False for submission
    )

    # === Step 5: Convert final masks to BraTS format
    print("Step 5: convert_back_BraTS_step_ratio_adaptive")
    convert_back_BraTS_step_ratio_adaptive(
        inference_folder=inference_folder,
        ensemble_code=ensemble_code,
        brats_final_inference=output_path
    )

    # === Step 6: Clean up
    print("Step 6: clean up the intermediate files in /output/tmp")
    cleanup_tmp = True  # set to False for debugging

    if cleanup_tmp:
        try:
            shutil.rmtree(tmp_path)
            print(f"Cleaned up temporary folder: {tmp_path}")
        except Exception as e:
            print(f"Failed to remove temporary folder {tmp_path}: {e}")

    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    infer(device=args.device)
