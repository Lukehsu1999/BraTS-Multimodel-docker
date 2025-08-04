#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
join = os.path.join

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

# nnUNet_raw = os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
# nnUNet_results = os.environ.get('nnUNet_results')

# base = '/media/volume1/khoa/BRATS25/7/Experiments/v01/nnUNetv2' # 150 cases
# base = '/media/volume1/khoa/BRATS25/7/Experiments/v02/nnUNetv2' # ALL cases with LABELS, nnunet v2.6.2
#OUT_BASE = '/media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2' # ALL cases with LABELS, nnunet v2.6.2
# IN_BASE = "/media/volume1/BraTS2025/7/nnUnet-Data"
# maybe_mkdir_p(base)

# https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
# nnUNet_raw = "/media/volume1/BraTS2025/7/nnUnet-Data" # 90% training data

# nnUNet_raw = "/media/volume1/BraTS2025/7/nnUnet-Data-Full-Andre" # 100% training data
# nnUNetv2_plan_and_preprocess -d 601 --verify_dataset_integrity -c 3d_fullres
# nnUNetv2_plan_experiment -d 601 -pl nnUNetPlannerResEncM
# nnUNetv2_plan_experiment -d 601 -pl nnUNetPlannerResEncL
# nnUNetv2_plan_experiment -d 601 -pl nnUNetPlannerResEncXL


# nnUNet_raw = "/media/volume1/BraTS2025/PseudoLabel_UnlabeledData" # pseudo label from training data wo GT
# nnUNetv2_plan_and_preprocess -d 602 --verify_dataset_integrity -c 3d_fullres
# nnUNetv2_plan_experiment -d 602 -pl nnUNetPlannerResEncM
# nnUNetv2_plan_experiment -d 602 -pl nnUNetPlannerResEncL
# nnUNetv2_plan_experiment -d 602 -pl nnUNetPlannerResEncXL

# nnUNet_raw = join(OUT_BASE, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = join(OUT_BASE, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
# nnUNet_results = join(OUT_BASE, 'nnUNet_results') # os.environ.get('nnUNet_results')

# Use env vars if available
nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

if nnUNet_raw is None:
    print("nnUNet_raw is not defined...")
if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined...")
if nnUNet_results is None:
    print("nnUNet_results is not defined...")

############## For Pseudo Label #######################

# # base = '/media/volume1/khoa/BRATS25/7/Experiments/v01/nnUNetv2' # 150 cases
# # base = '/media/volume1/khoa/BRATS25/7/Experiments/v02/nnUNetv2' # ALL cases with LABELS, nnunet v2.6.2
# OUT_BASE = '/media/volume1/khoa/BRATS25/7/Experiments/v03_nnunet211/nnUNetv2/PseudoLabel' # ALL cases with LABELS, nnunet v2.6.2
# # IN_BASE = "/media/volume1/BraTS2025/7/nnUnet-Data"
# # maybe_mkdir_p(base)

# # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
# # nnUNetv2_plan_and_preprocess -d 506 --verify_dataset_integrity -c 3d_fullres
# nnUNet_raw = f"{OUT_BASE}/nnUNet_raw_pseudolabel/"
# print(f"nnUNet_raw: {nnUNet_raw}")
# print(f"nnUNet_raw: {os.listdir(nnUNet_raw)}")

# # nnUNet_raw = join(OUT_BASE, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = join(OUT_BASE, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
# nnUNet_results = join(OUT_BASE, 'nnUNet_results') # os.environ.get('nnUNet_results')

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
