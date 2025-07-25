import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#导入qkeras 来支持Q模型

from qkeras.utils import _add_supported_quantized_objects
class Predictor:
    def __init__(self,model_path,max_depth=9.99547004699707):
        print("init model")
        co = {}
        _add_supported_quantized_objects(co)
        self.model = tf.keras.models.load_model(model_path,custom_objects=co)
        self.input_height = 256
        self.input_width = 256
        self.max_depth = max_depth
        print("model loaded successfully")

    def _preprocess(self,image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img,channels=3,expand_animations=False)
        img = tf.image.resize(img,[self.input_height,self.input_width])
        img = tf.cast(img,tf.float32) / 255.0
        img = tf.expand_dims(img,axis=0)
        return img
    
    def predict(self,image_path,output_path):
        print(f"loading img from:{image_path}")
        img = self._preprocess(image_path)

        prediction = self.model(img,training=False)

        prediction = prediction.numpy()

        prediction = prediction.squeeze()
        # --- 关键探针！ ---
        print("\n--- Prediction Analysis ---")
        print(f"Data type: {prediction.dtype}")
        print(f"Shape: {prediction.shape}")
        print(f"Min value: {np.min(prediction)}")
        print(f"Max value: {np.max(prediction)}")
        print(f"Mean value: {np.mean(prediction)}")
        print("---------------------------\n")
        # --- 探针结束 ---
            
            
        

        self._save(prediction,output_path)
        print("save to :f{output_path}")

    def _save(self,prediction,output_path):
        plt.imsave(output_path,prediction,cmap='magma')


#无菌输入 应该不会再使用
    def _preprocess_numpy(self, numpy_array):
        img = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
        img_resized = tf.image.resize(img, [self.input_height, self.input_width])
        img_normalized = img_resized / 255.0
        img_batched = tf.expand_dims(img_normalized, axis=0)
        return img_batched



    # 新增的“无菌预测”方法！
    def predict_from_numpy(self, numpy_array):
        """直接从一个Numpy数组进行预测，绕开所有文件I/O污染。"""
        print("\n--- Performing PURE prediction from Numpy array ---")
        processed_tensor = self._preprocess_numpy(numpy_array)
        
        prediction_tensor = self.model(processed_tensor, training=False)
        prediction_numpy = prediction_tensor.numpy()
        depth_map = prediction_numpy.squeeze()
        
        print(f"Prediction Min: {np.min(depth_map)}")
        print(f"Prediction Max: {np.max(depth_map)}")
        print(f"Prediction Mean: {np.mean(depth_map)}")
        
        return depth_map