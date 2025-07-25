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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


    # 实例化tuner
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective('val_mae',direction='min'),
        max_trials=50,
        executions_per_trial=1,
        directory='q_midas_tuning_v2',
        project_name='q_midas_small_v2'
    )

    tuner.search_space_summary()

    print("开启超参数搜索")

    tuner.search(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=40,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
    )

    print("搜索完成")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model = tuner.get_best_models(num_models=1)[0]
    final_quantization_blueprint = get_quantization_dictionary(best_model)
    save_quantization_dict("result/final_blueprint.json", best_model)
    pprint.pprint(final_quantization_blueprint)
    print_qmodel_summary(best_model)
    best_model.save('result/q_midas_small_best_v2.h5')
    print("最佳模型已保存到 result/q_midas_small_best_v2.h5")