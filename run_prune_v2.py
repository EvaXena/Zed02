#调用callbacks_v2.py中的回调函数 更加优化训练
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# 关键！我们现在需要把那个模型创建函数，直接传给回调函数！
from model.prune_midas_v1 import Midas_prune 
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import matplotlib.pyplot as plt

from callbacks_v2 import all_callbacks

def preprocess(image, depth):
    # 这个函数将在tf.data管道中被高效执行
    image_resized = tf.image.resize(image, [256, 256])
    depth_resized = tf.image.resize(depth, [256, 256])
    image_processed = tf.cast(image_resized, tf.float32) / 255.0
    depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707
    return image_processed, depth_processed

if __name__ == '__main__':
    # 不需要 mp.set_start_method 了

    model_name = "midas_prune_v6"
    
    print("Stage 1: Loading dataset paths...")
    with h5py.File('dataset/nyu_depth_v2_labeled.mat', 'r') as f:
        # 加载数据到内存
        images_all = np.transpose(f['images'][:], (0, 2, 3, 1))
        depths_all = np.expand_dims(f['depths'][:], axis=-1)
    
    print("Dataset loaded into memory. Creating data pipelines...")
    
    all_indices = np.arange(len(images_all))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.15, random_state=42
    )

    # 创建训练数据管道
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (images_all[train_indices], depths_all[train_indices])
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024) # 打乱数据
    train_dataset = train_dataset.batch(64) # 定义批次大小
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) # 并行预处理
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # 预取数据，让GPU无需等待

    # 创建验证数据管道
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (images_all[val_indices], depths_all[val_indices])
    )
    val_dataset = val_dataset.batch(64)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 后续的模型创建、编译、训练代码 ---
    print("stage 2: Model setup")
    # 我们先创建一个模型实例，用于训练
    model_to_train = Midas_prune()

    pruning_callbacks = all_callbacks(
        output_dir='result',
        model_name=model_name,
        pruning_model_fn=Midas_prune,  # 传入模型创建函数
        save_period=50,  # 每5个epoch保存一次
        stop_patience=200,  # 停止训练的耐心值
        lr_patience=20,  # 学习率调整的耐心值
        lr_factor=0.5  # 学习率调整的因子
    )

    print("start train")
    EPOCHS = 500

    history = model_to_train.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=pruning_callbacks
    )

    print("\nTraining complete. All necessary models have been saved periodically.")

    # --- 第八步：分析实验结果 ---
    print("Stage 6: Analyzing experimental results...")
    # (你需要先安装matplotlib: pip install matplotlib)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig('training_analysis.png')
    print("Analysis plot saved as 'training_analysis.png'.")
    plt.show()