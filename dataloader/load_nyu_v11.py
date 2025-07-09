# 在你的 nyu_dataloader.py 文件里，用这个新类替换掉旧的
import h5py
import numpy as np
import tensorflow as tf

class NYUv2Generator(tf.keras.utils.Sequence):
    """
    一个真正的、内存高效的数据生成器。
    它在需要时才从硬盘加载每一批数据。
    """
    def __init__(self, dataset_path, batch_size, indices, is_training=True):
        self.path = dataset_path
        self.batch_size = batch_size
        self.indices = indices # 接收训练集或验证集的索引
        self.is_training = is_training

        # 只打开文件句柄，不加载数据！
        self.h5_file = h5py.File(self.path, 'r')
        self.images_h5 = self.h5_file['images']
        self.depths_h5 = self.h5_file['depths']
        
        # 计算并存储最大深度值
        # 注意：这里为了获取max_depth还是会加载整个深度数据，但在实际生产中
        # 这个值通常是预先计算好并作为常量存储的。为了教学，我们暂时保留。
        # 如果这里仍然导致OOM，你需要预计算这个值并硬编码。
        self.max_depth = 9.99547004699707

    def __len__(self):
        """告诉Keras一个epoch有多少个批次"""
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """生成一个批次的数据"""
        # 计算当前批次需要用到的索引
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        # 使序号有序
        sorted_batch_indices = np.sort(batch_indices)
        
        # 从硬盘加载数据
        images_batch = np.transpose(self.images_h5[sorted_batch_indices], (0, 2, 3, 1))
        depths_batch = self.depths_h5[sorted_batch_indices]

        # 将Numpy数组转换为Tensor
        images_tensor = tf.convert_to_tensor(images_batch)
        depths_tensor = tf.convert_to_tensor(np.expand_dims(depths_batch, axis=-1))

        # 使用tf.image.resize进行缩放
        images_resized = tf.image.resize(images_tensor, [256, 256])
        depths_resized = tf.image.resize(depths_tensor, [256, 256])

        # 对缩放后的结果进行归一化
        images_processed = tf.cast(images_resized, tf.float32) / 255.0
        depths_processed = tf.cast(depths_resized, tf.float32) / self.max_depth

        # --- "黑匣子记录仪" ---
        # 这一行，是打破循环的唯一希望！
        print(f"DEBUGGER: Final shapes returned -> images: {images_processed.shape}, depths: {depths_processed.shape}")

        return images_processed, depths_processed

    def on_epoch_end(self):
        """在每个epoch结束后，如果是训练模式，就打乱索引"""
        if self.is_training:
            np.random.shuffle(self.indices)