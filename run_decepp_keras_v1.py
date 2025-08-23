# =================================================================
#           纯 Keras 版本的 Zero-DCE++ 训练脚本
# =================================================================
# - 已移除所有 HGQ 特有代码 (Imports, beta, Callbacks, 部署流程)
# - 使用您之前创建的 keras_zero_decepp_v1.py 文件来定义模型
# - 数据加载、预处理和核心自定义训练循环保持不变




#1e-4 全局变暗 应该缩小学习率 避免噪声影响 batchsize =  8
#1e-5 正在尝试 G通道全部归零 这扯不扯 batchsize =  8
#增大学习率为1e-4 batch_size扩大到64
#v2 全绿
#v3 修改l_col 依旧GB为1 R近0
#V4 加大l_col的权重 改为5 大部分黑色 其他红蓝绿
#v5 1e-4学习率 batchsize=8 尝试重现v1结果 命名为V4
#v6 使用flsea图像进行训练
# --- 修改: 清理 imports ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 确保这是脚本最先执行的几行代码之一

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- 修改: 导入纯 Keras 模型 ---
# 确保您已经将之前生成的 Keras 模型代码保存为 model/keras_zero_decepp_v1.py
from model.zero_decepp_v1 import *

# --- 图像预处理函数 (保持不变) ---
INPUT_SIZE = 512
def load_and_preprocess_image(path):
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img_resized = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img_processed = tf.cast(img_resized, tf.float32) / 255.0
    return img_processed

if __name__ == '__main__':
    # --- 超参数和路径配置 ---
    LR = 1e-4
    EPOCHS = 800
    BATCH_SIZE = 8 # 64 可能会导致显存不足，8 是一个更安全的值
    VAL_SPLIT = 0.15
    DATA_DIR = '/data2/home/clp/Documents/0709_Zed02/0709_Zed02_finished/Zed02/flsea_dataset_jpg'
    patience = 200
    # --- 修改: 更新输出路径并移除 beta ---
    os.makedirs('result', exist_ok=True)
    best_weights_path = 'result/best_keras_decepp_v6_weights.h5'
    final_model_path = "result/final_keras_decepp_v6.h5"

    # --- 数据加载和处理 (保持不变) ---
    print('Stage 1: Loading dataset paths')
    patterns = [os.path.join(DATA_DIR, '**/*.jpg'), os.path.join(DATA_DIR, '**/*.JPG'),os.path.join(DATA_DIR,'**/*.tiff')]
    all_file_paths = []
    # 使用 glob, 因为我们之前确认它比 list_files 在您的环境中更可靠
    import glob 
    for pattern in patterns:
        all_file_paths.extend(glob.glob(pattern, recursive=True))
    
    if not all_file_paths:
        raise ValueError(f"No images found in {DATA_DIR}. Please check the path.")
    
    np.random.shuffle(all_file_paths)
    
    val_size = int(len(all_file_paths) * VAL_SPLIT)
    train_paths = all_file_paths[val_size:]
    val_paths = all_file_paths[:val_size]
    print(f"Dataset created: {len(train_paths)} training images, {len(val_paths)} validation images")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- 训练准备 ---
    print('Stage 2: Preparing for training')
    
    # --- 修改: 实例化纯 Keras 模型 (不再需要 beta) ---
    model = Model(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), scale_factor=16) # 使用能被512整除的scale_factor
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.summary()

    # --- 修改: 移除 HGQ Callbacks ---
    
    # callbacks = [ResetMinMax(), FreeBOPs()] # 已移除
    # for cb in callbacks: # 已移除
    #     cb.set_model(model) # 已移除

    # --- 自定义训练循环 (核心逻辑保持不变) ---
    print('Stage 3: Starting custom training')
    best_val_loss = float('inf')
    wait_epochs = 0
    
    for epoch in range(EPOCHS):
        # --- 修改: 移除 cb.on_epoch_begin(epoch) ---
        
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()

        for image_batch in train_dataset:
            loss = train_step(model, image_batch, optimizer)
            train_loss_metric.update_state(loss)

        for image_batch in val_dataset:
            loss = val_step(model, image_batch)
            val_loss_metric.update_state(loss)

        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()

        # --- 修改: 移除 logs 字典和 cb.on_epoch_end ---
        # logs = {'loss': train_loss, 'val_loss': val_loss} # 已移除
        # for cb in callbacks: # 已移除
        #     cb.on_epoch_end(epoch, logs=logs) # 已移除

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 手动早停 (逻辑保持不变)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait_epochs = 0
            model.save_weights(best_weights_path)
            model.compile()
            model.save(final_model_path)
            print(f"  Validation loss improved. Saving best weights to {best_weights_path}")
        else:
            wait_epochs += 1
            if wait_epochs >= patience:
                print(f"  Validation loss did not improve for {patience} epochs. Stopping training.")
                break
    print("--- Training finished ---")

    # --- 修改: 移除并替换整个 HGQ 部署流程 ---

    # 1. 恢复最佳权重
    print(f"\nRestoring model with best weights from {best_weights_path} (Val Loss: {best_val_loss:.4f})")
    model.load_weights(best_weights_path)

    # 2. 保存最终的、可直接使用的 Keras 模型
    print(f"\nSaving final Keras model to {final_model_path}")
    model.save(final_model_path)
    print("Final model saved successfully.")