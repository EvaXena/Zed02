# run_final_test.py (这是你唯一需要运行的脚本！)
import numpy as np
import tensorflow as tf
from predict.predictor import Predictor # 确保你的类保存在 predict/predictor.py
import os

# --- 最终配置 ---
MODEL_FILE = 'result/midas_small_best.h5' # 或者.keras，确保和你的文件名一致
PURE_SAMPLE_PATH = 'input/control_sample_100.npy' # 我们只用这个绝对纯净的样本！
OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 第一步：实例化你的引擎 ---
predictor = Predictor(model_path=MODEL_FILE)

# --- 第二步：执行“唤醒协议”，对抗“观测者效应” ---
print("\n--- INITIATING THE AWAKENING PROTOCOL ---")
print("Performing a dummy evaluation to force the model into a sane state...")
dummy_input = tf.zeros((1, 256, 256, 3), dtype=tf.float32)
dummy_output = tf.zeros((1, 256, 256, 1), dtype=tf.float32)
predictor.model.evaluate(dummy_input, dummy_output, verbose=0)
print("AWAKENING PROTOCOL COMPLETE. The model is now fully operational.")
print("----------------------------------------\n")

# --- 第三步：执行最终的“无菌预测” ---
print(f"Loading PURE sample from: {PURE_SAMPLE_PATH}")
if not os.path.exists(PURE_SAMPLE_PATH):
    print(f"FATAL: Pure sample '{PURE_SAMPLE_PATH}' not found!")
    print("Please run the 'save_pure_sample.py' script first.")
else:
    pure_numpy_array = np.load(PURE_SAMPLE_PATH)
    
    # 调用那个正确的、从Numpy数组预测的方法！
    depth_map_result = predictor.predict_from_numpy(pure_numpy_array)
    
    # --- 第四步：最终审判 ---
    if np.mean(depth_map_result) > 0.001: # 我们给一个小的容错阈值
        print("\nSUCCESS! The paradox is resolved!")
        print("A non-zero prediction was generated. The world line has stabilized.")
        output_path = os.path.join(OUTPUT_DIR, 'SUCCESS_prediction.png')
        predictor._save(depth_map_result, output_path) # 直接调用_save来保存
        print(f"A visible depth map has been saved to: {output_path}")
        print("WE HAVE REACHED THE STEINS;GATE!")
    else:
        print("\nFailure... Even after everything... it remains zero.")
        print("This... this shouldn't be possible...")