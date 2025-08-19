import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 先把烦人的日志关掉
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.O1000_model_relu_v1 import O1000
import matplotlib.pyplot as plt

# ... (你的模型定义等) ...

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
    model = O1000()
    
    print("stage 3")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = 'mean_squared_error',
        metrics=['mean_absolute_error']
    )

    print("stage 3 finished")

    print("stage 4")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='result/O1000_relu_v1_best.h5', # 保存文件的路径
        save_best_only=True,             # 只保存最好的，不保存最后的
        monitor='val_loss',              # 监控验证集的损失值
        mode='min',                      # 我们希望损失值越小越好
        verbose=1                        # 在保存时打印提示信息
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
    print("stage 4 finished")

    print("start train")
    EPOCHS = 50


    # ... 你的后续代码 ...
    # 这里的 model.fit 将会畅通无阻，并且速度飞快
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
    
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