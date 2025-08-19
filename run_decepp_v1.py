import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 先把烦人的日志关掉
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model.hgq_zero_decepp_v1 import *
from HGQ import ResetMinMax, FreeBOPs
import tensorflow as tf
from HGQ import trace_minmax, to_proxy_model
import keras_tuner as kt
from model.q_midas_small_v2 import build_model
from qkeras.autoqkeras import *

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
import pprint
from HGQ import trace_minmax, to_proxy_model



INPUT_SIZE = 600


#图像类数据集 读取并预处理图像
def load_and_preprocess_image(path):
    #所有数据处理与读取全部用tf 确保兼容
    #tf.io.read_file读取文件
    img_raw = tf.io.read_file(path)
    #tf.io.decode_image解码图像
    img = tf.io.decode_image(img_raw,channels=3,expand_animations=False)
    #.set_shape设置数据形状
    img.set_shape([None,None,3])
    #tf.image.resize调整尺寸
    img_resized = tf.image.resize(img,[INPUT_SIZE,INPUT_SIZE])
    #tf.cast 改变张量内数据类型并归一化
    img_processed = tf.cast(img_resized,tf.float32) / 255.0
    return img_processed








if __name__ == '__main__':
    LR = 1e-4
    EPOCHS = 800
    BATCH_SIZE = 8
    VAL_SPILT = 0.15 #15%的数据用于验证
    DATA_DIR = '/data2/home/clp/Documents/0709_Zed02/0709_Zed02_finished/Zed02/decepp_train_dataset'
    OUTPUT_DIR = ''
    beta = 3e-6
    best_weights_path = 'result/best_decepp_v1.h5'
    proxy_model_path = "result/hgq_dce_proxy_model.h5"
    VAL_SPLIT = 0.15
    patience = 1000

    
     # --- 数据加载和处理 (*** 这是被修改的部分 ***) ---
    print('Stage 1: Loading dataset paths using Python glob')
    
    # 1. 使用 glob 递归搜索文件，我们已经证明这是有效的
    patterns = [
        os.path.join(DATA_DIR, '**/*.jpg'),
        os.path.join(DATA_DIR, '**/*.JPG')
    ]
    all_file_paths = []
    for pattern in patterns:
        all_file_paths.extend(glob.glob(pattern, recursive=True))

    # 随机打乱文件路径列表
    np.random.shuffle(all_file_paths)

    if not all_file_paths:
        raise ValueError(f"No images found in {DATA_DIR} using glob. Please double-check the path and file structure.")
    
    print(f"Glob found {len(all_file_paths)} image files.")

    # 2. 在 Python 列表中划分训练集和验证集
    val_size = int(len(all_file_paths) * VAL_SPLIT)
    train_paths = all_file_paths[val_size:]
    val_paths = all_file_paths[:val_size]
    
    print(f"Dataset created: {len(train_paths)} training images, {len(val_paths)} validation images")

    # 3. 使用 from_tensor_slices 从路径列表创建数据集
    # 这是关键的修改！
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)



    #第三步 创建高效tf.data管道
    #创建训练数据管道

    #首先dataset.map(func,num_parallel_calls) 调用读取函数 
    #将func应用到train_set内的每一个文件上
    #load_and_preprocess输入一个path 而train_set内部都是path
    train_dataset = train_dataset.map(load_and_preprocess_image,num_parallel_calls = tf.data.AUTOTUNE)
    #.shuffle维护一个大小为buffer_size的池来随机打乱输入数据 一般比batch_size大得多
    train_dataset = train_dataset.shuffle(buffer_size = 1024)
    #dataset.batch()划分BATCH_SIZE 在池中将单个样本合成为一个batch
    train_dataset = train_dataset.batch(BATCH_SIZE)
    #.pretetch在GPU计算时 让CPU提前预取数据
    train_dataset = train_dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    print('stage2 start pre_train setting')


    #实例化模型 优化器 callback
    model = Model(beta = beta)
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.summary()

    
    callbacks = [ResetMinMax(),FreeBOPs()]
    #手动将模型与callbacks关联
    for cb in callbacks:
        cb.set_model(model)

    print('stage3 start custom training')

    best_val_loss = float('inf')
    wait_epochs = 0
    
    for epoch in range(EPOCHS):
        #手动启动callback
        for cb in callbacks:
            cb.on_epoch_begin(epoch)
        
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()

        for image_batch in train_dataset:
            loss = train_step(model,image_batch,optimizer)
            train_loss_metric.update_state(loss)

        for image_batch in val_dataset:
            loss = val_step(model,image_batch)
            val_loss_metric.update_state(loss)

        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()

        #logs字典用来给callback传递所需值
        logs = {'loss':train_loss,'val_loss':val_loss}

        for cb in callbacks:
            cb.on_epoch_end(epoch,logs=logs)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        #手动早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait_epochs = 0
            model.save_weights(best_weights_path)
            print(f"  Validation loss improved. Saving best weights to {best_weights_path}")
        else:
            wait_epochs += 1
            if wait_epochs >= patience:
                print(f"  Validation loss did not improve for {patience} epochs. Stopping training.")
                break
    print("--- 训练完成 ---")

    # --- 4. 恢复最佳权重 ---
    print(f"Restoring model with best weights from {best_weights_path} (Val Loss: {best_val_loss:.4f})")
    model.load_weights(best_weights_path)

    # --- 5. 执行 HGQ 部署流程 ---
    print("\n--- 正在固化模型的数值范围 (Trace Min/Max)... ---")
    # 从训练集中取出一批样本用于固化
    sample_batch = next(iter(train_dataset.take(1)))
    trace_minmax(model, sample_batch.numpy())
    print("数值范围已固化。")

    print("\n--- 正在生成代理模型 (Proxy Model)... ---")
    proxy_model = to_proxy_model(model)
    print("代理模型已生成！这是一个纯粹的、静态的QKeras模型。")

    # --- 6. 保存最终可用于 HLS4ML 的模型 ---
    
    proxy_model.save(proxy_model_path)
    print(f"最终的、可用于hls4ml的代理模型已保存至 {proxy_model_path}")