# run_comparison.py (The "Pre-processing Aware" Final Version)

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

from comparator import Comparator 

# --- 数据准备阶段 ---
print("--- Data Preparation Stage ---")
with h5py.File('dataset/nyu_depth_v2_labeled.mat', 'r') as f:
    images_all = np.transpose(f['images'][:], (0, 2, 3, 1))
    depths_all = np.expand_dims(f['depths'][:], axis=-1)

# 逻辑划分
train_val_indices, test_indices = train_test_split(
    np.arange(len(images_all)), test_size=0.15, random_state=42)

X_test_raw = images_all[test_indices]
y_test_raw = depths_all[test_indices]
print(f"Raw test set prepared. Shape: {X_test_raw.shape}")

# --- 关键修正！我们必须在这里创建预处理过的数据管道！ ---
print("Creating pre-processed test data pipeline...")

def preprocess(image, depth):
    image_resized = tf.image.resize(image, [256, 256])
    depth_resized = tf.image.resize(depth, [256, 256])
    image_processed = tf.cast(image_resized, tf.float32) / 255.0
    depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707
    return image_processed, depth_processed

# 创建一个tf.data.Dataset，这才是正确的喂给evaluate的数据格式
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_raw, y_test_raw))
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32) # 使用一个合适的batch_size
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("--- Data Preparation Complete ---\n")


if __name__ == '__main__':
    # --- 实验配置 ---
    BASELINE_MODEL_PATH = 'result/midas_small_best_v2.h5'
    PRUNED_MODEL_PATH = 'result/final_pruned_model_v1.h5' # 确保这是一个真的被剪枝过的模型！

    # --- 实例化“审判者” ---
    comparator = Comparator(
        baseline_model_path=BASELINE_MODEL_PATH,
        compressed_model_path=PRUNED_MODEL_PATH
    )
    
    # --- 开始审判 ---
    # 修改 comparator.py 里的 compare_regression_performance 方法，
    # 让它接收一个 tf.data.Dataset 对象！
    
    (comparator
        .compare_model_size()
        .compare_flops()
        .compare_regression_performance(test_dataset) # 这里的调用现在和上面的定义完全匹配！
        .generate_summary_report()
    )

    print("\nComparison complete. This is the choice of Steins;Gate.")