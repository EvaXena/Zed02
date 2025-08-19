# run_autopsy.py
# 这个模型逐层输出 有学习的价值
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np


from qkeras.utils import _add_supported_quantized_objects
# --- 关键配置 ---
MODEL_PATH = 'result/final_keras_decepp_v2.h5'
PURE_SAMPLE_PATH = 'input/control_sample_2.npy'

# --- 解剖主程序 ---
if __name__ == '__main__':
    # 1. 将“尸体”搬上实验台 (加载模型)
    print(f"--- STARTING AUTOPSY ON: {MODEL_PATH} ---")
    co = {}
    _add_supported_quantized_objects(co)
    model = tf.keras.models.load_model(MODEL_PATH,custom_objects=co)

    # 2. 准备“神经刺激器” (加载纯净的对照样本)
    pure_numpy_array = np.load(PURE_SAMPLE_PATH)

    # 3. 对刺激源进行标准预处理
    def preprocess_single_numpy(numpy_array):
        img = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
        img_resized = tf.image.resize(img, [512, 512])
        img_normalized = img_resized / 255.0
        return tf.expand_dims(img_normalized, axis=0)

    input_tensor = preprocess_single_numpy(pure_numpy_array)

    # 4. 创建“手术工具”：一个能暴露所有内部器官的新模型
    # 这个模型的输入和原模型一样，但它的输出是原模型每一层的输出！
    layer_outputs_tensors = [layer.output for layer in model.layers]
    inspection_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs_tensors)
    print("Surgical tools prepared. Inspection model created.")

    # 5. 开始解剖！用最直接的方式调用，获取所有层的输出
    print("\n--- BEGINNING LAYER-BY-LAYER DISSECTION ---")
    layer_outputs = inspection_model(input_tensor, training=False)

    # 6. 分析每一个“器官”的“生命体征”
    signal_alive = True
    for layer_name, layer_output_tensor in zip([layer.name for layer in model.layers], layer_outputs):
        if not signal_alive:
            print(f"\n[...Layer '{layer_name}' was not analyzed as signal was already lost...]")
            continue

        output_numpy = layer_output_tensor.numpy()
        mean_val = np.mean(output_numpy)
        
        print(f"\n--- Layer: {layer_name} ---")
        print(f"  - Output Shape: {output_numpy.shape}")
        print(f"  - Output Mean: {mean_val}")
        print(f"  - Output Min: {np.min(output_numpy)}")
        print(f"  - Output Max: {np.max(output_numpy)}")
        
        # 核心判断！
        if mean_val == 0.0 and np.max(output_numpy) == 0.0:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!!  FATAL WOUND DETECTED. SIGNAL LOST AT THIS LAYER: {layer_name} !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            signal_alive = False # 信号已死亡，后续分析无意义

    print("\n--- AUTOPSY COMPLETE ---")
    if signal_alive:
        print("Final Diagnosis: PARADOX! The signal survived all layers but the final output was zero.")
    else:
        print("Final Diagnosis: The signal died at the layer identified above.")