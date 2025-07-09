from sklearn.model_selection import train_test_split
from dataloader.load_nyu_v11 import NYUv2Generator# 导入新的类
from model.midas_small_v1 import Midas_small
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy('mixed_float16')#采用16位训练

if __name__ == '__main__':
    # --- 第一步：准备索引 ---
    # 我们仍然需要知道总共有多少样本，但只加载索引，不加载数据
    with h5py.File('dataset/nyu_depth_v2_labeled.mat', 'r') as f:
        num_samples = f['images'].shape[0]
    
    all_indices = np.arange(num_samples)

    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.15, random_state=42
    )

    # --- 第二步：实例化生成器 ---
    train_generator = NYUv2Generator(
        dataset_path='dataset/nyu_depth_v2_labeled.mat',
        batch_size=4,
        indices=train_indices,
        is_training=True
    )
    val_generator = NYUv2Generator(
        dataset_path='dataset/nyu_depth_v2_labeled.mat',
        batch_size=4,
        indices=val_indices,
        is_training=False
    )
    
    # ... 你的模型创建、编译代码 ...
    
    print("stage 2")

    model = Midas_small()
    model.summary()

    print("stage 2 finished")

    print("stage 3")

    model.compile(
        optimizer='adam',
        loss = 'mean_squared_error',
        metrics=['mean_absolute_error']
    )

    print("stage 3 finished")

    print("stage 4")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='result/midas_small_best.keras', # 保存文件的路径
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
    EPOCHS = 250

    # --- 第七步：启动训练！ ---
    # .fit() 方法原生支持Sequence生成器！
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
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