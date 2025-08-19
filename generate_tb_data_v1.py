import tensorflow as tf
import numpy as np
import hls4ml
import os
from PIL import Image

# -----------------------------------------------------------------
#           第一步：召唤“未来道具”的核心 (The Core)
# -----------------------------------------------------------------

# 导入你的自定义层，这是绝对的前提
from model.layers.tanhlu import Tanhlu
# 导入QKeras的工具
from qkeras.utils import _add_supported_quantized_objects

# 准备自定义对象字典，这直接从你的Predictor类里抄过来
print("Preparing custom objects for model loading...")
custom_objects = {'Tanhlu': Tanhlu}
_add_supported_quantized_objects(custom_objects)

# 加载你训练好的Keras模型
print("Loading trained Keras model...")
keras_model = tf.keras.models.load_model('O1000_relu_v1_best.h5', custom_objects=custom_objects)
keras_model.summary()

# 转换成hls_model
print("Converting Keras model to HLS model...")
config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
# ... 你可以在这里进一步修改config ...
hls_model = hls4ml.converters.convert_from_keras_model(
    keras_model,
    hls_config=config,
    output_dir='hls_final_project',
    backend='Vivado',
    part='your_fpga_part'
)

# -----------------------------------------------------------------
#           第二步：准备“观测样本” (The Samples)
# -----------------------------------------------------------------

# 定义测试图片的来源和数量
TEST_IMAGE_DIR = 'test_images/'
NUM_SAMPLES = 10 # 我们只用10张图片来生成测试台，太多会很慢

# 从你的Predictor类中，把预处理的逻辑“借”过来
# 我们把它变成一个独立的函数
def preprocess_image(image_path, input_height=256, input_width=256):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [input_height, input_width])
    img = tf.cast(img, tf.float32) / 255.0
    # 注意：这里我们先不加batch维度，因为要将它们堆叠起来
    return img

print(f"\nLoading and preprocessing {NUM_SAMPLES} test images...")
image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR)][:NUM_SAMPLES]

# 循环处理每一张图片，并将它们收集到一个列表里
processed_images_list = []
for img_path in image_files:
    processed_img = preprocess_image(img_path)
    processed_images_list.append(processed_img)

# 将图片列表堆叠成一个大的NumPy数组，形成一个“批次”
# 这就是我们需要的 x_test_batch
x_test_batch = np.array(processed_images_list)
print(f"Created x_test_batch with shape: {x_test_batch.shape}")


# -----------------------------------------------------------------
#           第三步：生成“黄金标准答案”
# -----------------------------------------------------------------
print("Generating Keras predictions to be used as golden reference...")
y_keras_predictions = keras_model.predict(x_test_batch)
print(f"Generated y_keras_predictions with shape: {y_keras_predictions.shape}")


# -----------------------------------------------------------------
#           第四步：执行最终咒语，生成tb_data！
# -----------------------------------------------------------------
print("Writing test bench data (tb_data)...")
hls_model.write_test_bench(x_test_batch, y_keras_predictions)

print("\n--- Mission Complete ---")
print("tb_data directory has been successfully generated in 'hls_final_project/'.")
print("You can now run C-Simulation using hls_model.build(csim=True)")