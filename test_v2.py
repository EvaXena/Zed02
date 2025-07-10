# validate_saved_model.py
import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os

# --- 关键配置 ---
MODEL_PATH = 'result/midas_small_best.h5'
H5_PATH = 'dataset/nyu_depth_v2_labeled.mat'

# --- 预处理函数 (必须和训练时完全一致!) ---
def preprocess(image, depth):
    image_resized = tf.image.resize(image, [256, 256])
    depth_resized = tf.image.resize(depth, [256, 256])
    image_processed = tf.cast(image_resized, tf.float32) / 255.0
    depth_processed = tf.cast(depth_resized, tf.float32) / 9.99547004699707
    return image_processed, depth_processed

# --- 主执行区 ---
if __name__ == '__main__':
    # 1. 加载模型 (进行灵魂召唤)
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("FATAL ERROR: Model file does not exist!")
        exit()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully. Now performing sanity checks...")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model. Error: {e}")
        exit()

    # 2. 准备一小批“绝对标准”的测试数据
    print("Preparing a small, standard test dataset...")
    with h5py.File(H5_PATH, 'r') as f:
        images_all = np.transpose(f['images'][:], (0, 2, 3, 1))
        depths_all = np.expand_dims(f['depths'][:], axis=-1)
    
    # 我们只需要一小部分数据来进行快速验证
    _, test_indices = train_test_split(np.arange(len(images_all)), test_size=0.2, random_state=42)
    X_test_sample = images_all[test_indices[:32]] # 只取32张图
    y_test_sample = depths_all[test_indices[:32]]

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_sample, y_test_sample))
    test_dataset = test_dataset.batch(4).map(preprocess).prefetch(tf.data.AUTOTUNE)

    # 3. 进行官方的“灵魂拷问”第一式：model.evaluate()
    # 这是最权威的测试，它会计算模型在标准数据上的损失值
    print("\n--- Performing Official Evaluation (model.evaluate) ---")
    results = model.evaluate(test_dataset, verbose=0)
    print(f"Evaluation Loss: {results[0]}")
    print(f"Evaluation MAE: {results[1]}")
    print("------------------------------------------------------")
    if np.isnan(results[0]) or results[1] == 0:
        print("DIAGNOSIS: Evaluation results are nonsensical. The model is likely corrupted or was never trained properly.")

    # 4. 进行官方的“灵魂拷问”第二式：model.predict()
    # 我们直接在这里预测，看看输出是不是0
    print("\n--- Performing Manual Prediction Check ---")
    for images, _ in test_dataset.take(1):
        prediction = model.predict(images)
        print(f"Prediction Min value: {np.min(prediction)}")
        print(f"Prediction Max value: {np.max(prediction)}")
        print(f"Prediction Mean value: {np.mean(prediction)}")
        if np.mean(prediction) == 0:
            print("DIAGNOSIS: Prediction is all zeros. The model is effectively dead.")
        else:
            print("DIAGNOSIS: Prediction seems to produce non-zero values. The issue might be in your original inference script.")
    print("------------------------------------------")