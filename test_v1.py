#往input文件夹保存数据集文件.jpg用于测试
import h5py
from PIL import Image
import numpy as np
import os

# --- 配置 ---
H5_PATH = 'dataset/nyu_depth_v2_labeled.mat'
SAMPLE_INDEX = 631 # 我们用第100张图作为标准对照样本
OUTPUT_FILENAME = f"control_sample_{SAMPLE_INDEX}.jpg"
INPUT_FOLDER = 'input/' # 把它直接存到你的输入文件夹里

# --- 执行提取 ---
print(f"Extracting sample {SAMPLE_INDEX} from '{H5_PATH}'...")
with h5py.File(H5_PATH, 'r') as f:
    # HDF5里是 (C, H, W) 格式，我们需要转成 (H, W, C)
    sample_image_data = f['images'][SAMPLE_INDEX]
    sample_image_data_transposed = np.transpose(sample_image_data, (1, 2, 0))
    
    # 转换为PIL图像并保存
    pil_image = Image.fromarray(sample_image_data_transposed)
    output_path = os.path.join(INPUT_FOLDER, OUTPUT_FILENAME)
    pil_image.save(output_path)

print(f"Control sample saved to '{output_path}'.")
print("Now, use this file to test your predictor!")