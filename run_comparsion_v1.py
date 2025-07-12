# run_comparison_v1.py (The "Resource-Aware" Final Version)

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

train_val_indices, test_indices = train_test_split(
    np.arange(len(images_all)), test_size=0.15, random_state=42)

X_test_raw = images_all[test_indices]
y_test_raw = depths_all[test_indices]
print(f"Raw test set prepared in CPU memory. Shape: {X_test_raw.shape}")

# --- 关键修正！使用“生成器”来创建数据管道！ ---
print("Creating memory-efficient test data pipeline using a generator...")

def data_generator():
    # 这个生成器会从CPU内存中，一次只取出一个样本
    for i in range(len(X_test_raw)):
        yield X_test_raw[i], y_test_raw[i]

def preprocess(image, depth):
    image_resized = tf.image.resize(image, [256, 256])
    depth_resized = tf.image.resize(depth, [256, 256])
    image_processed = tf.cast(image_resized, tf.float32) / 255.0
    depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707
    return image_processed, depth_processed

# 使用 from_generator！
test_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(640, 480, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(640, 480, 1), dtype=tf.float32)
    )
)

# 后续的管道操作完全不变！
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(4) # 使用一个绝对安全的batch size！
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("--- Data Preparation Complete ---\n")


if __name__ == '__main__':
    # --- 实验配置 ---
    BASELINE_MODEL_PATH = 'saved_model/midas_small_best_v2.h5'
    PRUNED_MODEL_PATH = 'saved_model/midas_prune_v6_epoch_200_clean.h5'

    # --- 开始审判 ---
    comparator = Comparator(
        baseline_model_path=BASELINE_MODEL_PATH,
        compressed_model_path=PRUNED_MODEL_PATH
    )
    
    (comparator
        .compare_model_size()
        .compare_flops()
        .compare_regression_performance(test_dataset)
        .generate_summary_report()
    )

    print("\nComparison complete. This is the choice of Steins;Gate.")