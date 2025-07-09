#v13是专门在服务器上运行的高并行度代码
#用于快速产出完善的NYU数据集代码进行下一步实验
# run_midas_v12.py (修改版)
#修改学习率与低精度权重来避免nan
#使用python来调取 手动设置 效率低 GPU基本没跑

# 你的训练流程是这样的：

#     你那32个CPU“克隆分身”手忙脚乱地准备下一批次的数据（从内存里切片、做一些预处理等）。

#     这个过程非常缓慢，需要花费好几秒钟。在这期间，你所有的CPU核心都在哀嚎。

#     当一小批数据终于准备好后，被送到GPU。

#     RTX 4090的算力是何等恐怖？它只用了0.001秒就完成了这批数据的计算。

#     然后，GPU重新进入长达数秒的、漫长的等待状态，等待你那群慢吞吞的CPU分身准备好下一批数据。

# 所以，不是“打印日志拖慢了训练”，也不是“创造分身花了太久”，而是你的GPU大部分时间都在罢工！你看到日志之间漫长的间隔，正是GPU在等待CPU“投喂”食物的空闲时间！



from sklearn.model_selection import train_test_split
# 不再需要 NYUv2Generator 了，我们直接在这里处理
from model.midas_small_v1 import Midas_small
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp



# 为了在多进程中安全，定义一个简单的内存Generator
class InMemoryGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, depths, batch_size, is_training=True):
        self.images, self.depths = images, depths
        self.batch_size = batch_size
        self.is_training = is_training
        # 在初始化时就生成索引
        self.indices = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # 直接从内存中的Numpy数组切片，非常快！
        return self.images[batch_indices], self.depths[batch_indices]

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # --- 核心改动：在这里一次性加载所有数据到内存 ---
    print("Stage 1: Loading entire dataset into memory... (This may take a while)")
    with h5py.File('dataset/nyu_depth_v2_labeled.mat', 'r') as f:
        # 加载并立即处理，然后文件句柄就会被关闭
        images_all = np.transpose(f['images'][:], (0, 2, 3, 1))
        depths_all = np.expand_dims(f['depths'][:], axis=-1)
    
    print("Dataset loaded. Resizing and normalizing...")
    # 使用 tf.data 来利用GPU进行快速的预处理
    dataset = tf.data.Dataset.from_tensor_slices((images_all, depths_all))
    
    def preprocess(image, depth):
        image_resized = tf.image.resize(image, [256, 256])
        depth_resized = tf.image.resize(depth, [256, 256])
        image_processed = tf.cast(image_resized, tf.float32) / 255.0
        depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707 # 使用已知最大值
        return image_processed, depth_processed

    # 使用 .batch().map() 可以非常高效地处理数据
    # AUTOTUNE可以自动调整并行调用的数量
    dataset = dataset.batch(64).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # 将GPU处理完的数据拉回CPU，作为Numpy数组
    # 注意：这里会消耗大量内存！
    processed_images = np.concatenate([x for x, y in dataset])
    processed_depths = np.concatenate([y for x, y in dataset])
    
    print(f"Preprocessing finished. Final tensor shapes in memory: images={processed_images.shape}, depths={processed_depths.shape}")

    # --- 数据准备完毕，划分训练集和验证集 ---
    all_indices = np.arange(len(processed_images))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.15, random_state=42
    )
    
    train_generator = InMemoryGenerator(
        images=processed_images[train_indices],
        depths=processed_depths[train_indices],
        batch_size=4,
        is_training=True
    )
    val_generator = InMemoryGenerator(
        images=processed_images[val_indices],
        depths=processed_depths[val_indices],
        batch_size=4,
        is_training=False
    )
    
    # --- 后续的模型创建、编译、训练代码完全不变 ---
    print("stage 2")
    model = Midas_small()

    model.summary()

    print("stage 2 finished")

    print("stage 3")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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


    # ... 你的后续代码 ...
    # 这里的 model.fit 将会畅通无阻，并且速度飞快
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        workers=32, # 现在可以安全地使用多进程了
        use_multiprocessing=True,
        max_queue_size=20
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