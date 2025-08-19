from model.hgq_midas_small_v2 import build_hgq_model
from HGQ import ResetMinMax, FreeBOPs
import tensorflow as tf
from HGQ import trace_minmax, to_proxy_model
import keras_tuner as kt
import qkeras
from model.q_midas_small_v2 import build_model
from qkeras.autoqkeras import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 先把烦人的日志关掉
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pprint
from HGQ import trace_minmax, to_proxy_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def preprocess(image, depth):
    # 这个函数将在tf.data管道中被高效执行
    image_resized = tf.image.resize(image, [256, 256])
    depth_resized = tf.image.resize(depth, [256, 256])
    image_processed = tf.cast(image_resized, tf.float32) / 255.0
    depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707
    return image_processed, depth_processed



if __name__ == '__main__':
    # 不需要 mp.set_start_method 了

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
    train_dataset = train_dataset.batch(32) # 定义批次大小
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) # 并行预处理
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # 预取数据，让GPU无需等待

    # 创建验证数据管道
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (images_all[val_indices], depths_all[val_indices])
    )
    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    #设定参数 构建模型

    BETA_VALUE = 3e-6 

    print(f"正在使用 beta = {BETA_VALUE} 构建HGQ模型...")
    model = build_hgq_model(beta=BETA_VALUE)

    #设定callback 此callback为hqg独有callback

    callbacks = [
        ResetMinMax(),
        FreeBOPs(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        ]
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # 可以从一个标准的学习率开始
    loss_function = 'mean_squared_error'
    metrics = ['mae']

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # --- 开始训练！---
    # HGQ的作者建议使用大的batch_size和长的epochs，以获得稳定的结果
    EPOCHS = 200
    BATCH_SIZE = 1 # 或者 8192, 16384，取决于你的GPU显存

    print("--- 开始HGQ模型的训练仪式 ---")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    # --- 保存训练好的、但还不能直接使用的“原生HGQ模型” ---
    model.save("hgq_trained_model.h5")
    print("原生HGQ模型训练完成并已保存。")

    print("\n--- 正在固化模型的数值范围 (Trace Min/Max)... ---")
    # 注意：这里需要传入Numpy数组，而不是tf.data.Dataset
    # 你需要从你的训练数据集中，取出一部分样本
    (X_train_sample, y_train_sample) = next(iter(train_dataset.batch(500))) # 取10000个样本
    trace_minmax(model, X_train_sample.numpy())
    print("数值范围已固化。")

    # --- 翻译：创造那个最终的、可以被hls4ml使用的“代理模型” ---
    print("\n--- 正在生成代理模型 (Proxy Model)... ---")
    proxy_model = to_proxy_model(model)
    print("代理模型已生成！这是一个纯粹的、静态的QKeras模型。")

    # --- 保存我们真正的、最终的“未来道具”！---
    proxy_model.save("hgq_proxy_model.h5")
    print("最终的、可用于hls4ml的代理模型已保存至 hgq_proxy_model.h5")