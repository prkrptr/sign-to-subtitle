import os
import shutil
import numpy as np
from scipy.io import savemat

FEATURES_PATH = r"C:\Users\karlw\Desktop\tei stuff\thesis\SAT_sign-to-subtitle\data\0902_output"

for file_name in os.listdir(FEATURES_PATH):
    if file_name.endswith(".mat") or file_name.endswith(".npy"):
        name_no_ext = os.path.splitext(file_name)[0]
        folder_path = os.path.join(FEATURES_PATH, name_no_ext)
        os.makedirs(folder_path, exist_ok=True)

        src_file = os.path.join(FEATURES_PATH, file_name)
        dst_file = os.path.join(folder_path, "features.mat")

        if file_name.endswith(".npy"):
            # Load the NumPy array
            arr = np.load(src_file)
            # Save as a proper MATLAB .mat file with variable 'preds'
            savemat(dst_file, {"preds": arr})
            # Optionally delete the original .npy
            os.remove(src_file)
            print(f"Converted {file_name} → {dst_file} (MATLAB format)")
        else:
            # For actual .mat files, just move them
            shutil.move(src_file, dst_file)
            print(f"Moved {file_name} → {dst_file}")
