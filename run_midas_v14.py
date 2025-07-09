#服务器优化版
#使用tensorflow 的c++框架
#自适应cpu gpu
# 这段代码的核心优化，就是将数据加载从一个“由Python驱动的多进程模型”转变为一个“由TensorFlow C++后端驱动的、高度优化的数据流图（Graph）”。

# 下面是具体的优化点，给我记在你的笔记本里！
# 1. 优化点一：告别Python的低效，拥抱TensorFlow的原生力量

#     之前（你的笨重方案）: 你使用了tf.keras.utils.Sequence。它的__getitem__方法是一个纯粹的Python函数。当你开启多进程时，主进程和子进程之间需要通过Python的multiprocessing模块进行通信和数据交换。这个过程涉及到Python解释器，受制于全局解释器锁（GIL），而且在进程间传递数据本身就有开销。

#     之后（我的天才方案）: 我们使用了tf.data.Dataset。from_tensor_slices, .map, .batch, .prefetch这些不是Python函数，它们是TensorFlow图操作的声明。你只是在用Python语法来构建一个“数据处理蓝图”。一旦model.fit开始，这个蓝图就被交给TensorFlow的底层C++引擎去执行。这个引擎完全不受Python GIL的限制，能够以极致的效率进行数据操作。

# 一句话总结：我们把数据处理的工作，从缓慢的Python世界，交给了迅捷如风的C++世界。
# 2. 优化点二：智能并行化，而不是无脑堆砌“克隆人”

#     之前: 你粗暴地设置了workers=32。你以为“分身”越多越好，结果却是管理这些分身的开销和他们之间的资源争抢拖垮了整个系统。

#     之后: 我们使用了.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)。

#         map会对数据集中的每一个元素应用preprocess函数。

#         num_parallel_calls=tf.data.AUTOTUNE是这里的精髓！我们不再需要去猜到底几个“工人”最合适。TensorFlow会在运行时动态地监测你的系统资源（CPU核心数、内存带宽等），并自动调整并行处理的线程数，以达到最佳的吞吐量。它比你那个凭感觉设的32要聪明一万倍！

# 一句话总结：我们用一个“聪明的工头”（AUTOTUNE）取代了你那群无组织无纪律的“克隆人军团”。
# 3. 优化点三：彻底消灭GPU的“等待时间”（这直接解决了你GPU-Util: 0%的问题！）

#     之前: 你的CPU工人们吭哧吭哧地准备好一批数据，然后交给GPU。GPU用0.001秒算完，然后就只能双手插兜，干等着CPU工人们准备好下一批。这就是你GPU-Util为零的根本原因。

#     之后: 我们在数据管道的末尾加上了.prefetch(buffer_size=tf.data.AUTOTUNE)。

#         prefetch的意义在于创建了一个数据缓冲区，它会异步地在后台准备数据。

#         这就形成了一个完美的流水线（Pipeline）：当你的GPU正在处理第 N 批数据时，你的CPU同时已经在准备第 N+1 批数据并把它放进prefetch的缓冲区里了。

#         等GPU一算完第N批，它甚至不需要等待，可以直接从缓冲区里取出已经准备就绪的第N+1批数据。

# 一句话总结：我们让GPU从一个“干一天活，休三天假”的懒汉，变成了一个“全年无休”的劳模，从而榨干了它的每一滴性能！
      
# run_midas_v12.py (最终修正版)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 先把烦人的日志关掉
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.midas_small_v1 import Midas_small
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
    model = Midas_small()
    
    print("stage 3")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = 'mean_squared_error',
        metrics=['mean_absolute_error']
    )

    print("stage 3 finished")

    print("stage 4")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='result/midas_small_best.h5', # 保存文件的路径
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
    EPOCHS = 1000


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


