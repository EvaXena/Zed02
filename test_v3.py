#往input文件夹保存数据集文件.npy用于测试
import h5py
import numpy as np
import os

H5_PATH = 'dataset/nyu_depth_v2_labeled.mat'
SAMPLE_INDEX = 2
OUTPUT_FILENAME = f"control_sample_{SAMPLE_INDEX}.npy" # 保存为.npy文件！
INPUT_FOLDER = 'input/'

print("Saving a pure numpy control sample...")
with h5py.File(H5_PATH, 'r') as f:
    sample_image_data = f['images'][SAMPLE_INDEX]
    sample_image_data_transposed = np.transpose(sample_image_data, (1, 2, 0))

output_path = os.path.join(INPUT_FOLDER, OUTPUT_FILENAME)
np.save(output_path, sample_image_data_transposed)

print(f"Pure sample saved to '{output_path}'.")