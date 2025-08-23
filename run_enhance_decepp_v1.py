import os
# 确保 os.environ 设置在 tensorflow 导入之前或之后立即执行，以避免 GPU 问题
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 指定使用的 GPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# 1. 从您的模型文件中导入 Model 定义和 enhance 函数
#    确保路径和文件名正确
from model.zero_decepp_v1 import *

# ==============================================================================
#                             --- 配置区 ---
#                 在使用前，请修改这里的路径和参数
# ==============================================================================

# 模型权重文件的路径
MODEL_WEIGHTS_PATH = 'result/best_keras_decepp_v4_weights.h5'

# 您想要测试的输入图片的路径
INPUT_IMAGE_PATH = 'decepp_test_images/low_light_image.jpg' # TODO: 替换为您自己的图片路径

# 模型训练时使用的参数 (必须与训练时完全一致)
INPUT_SIZE = 512
SCALE_FACTOR = 16 # 我们确定了 16 是能被 512 整除的最佳值

# ==============================================================================

def load_and_preprocess_image(path):
    """
    加载、解码、调整尺寸并归一化图像，与训练时保持一致。
    """
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img_resized = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img_processed = tf.cast(img_resized, tf.float32) / 255.0
    return img_processed

def visualize(model, original_image_path):
    """
    执行完整的预测和可视化流程。
    """
    print(f"--- 正在处理图片: {original_image_path} ---")

    # 1. 加载并预处理原始图像
    original_image_processed = load_and_preprocess_image(original_image_path)
    print(f"图像已预处理为尺寸: {original_image_processed.shape}")

    # 2. 为模型预测准备输入张量 (添加一个批次维度)
    input_tensor = tf.expand_dims(original_image_processed, axis=0)
    print(f"输入模型的张量形状: {input_tensor.shape}")

    # 3. 通过模型进行前向传播，获取曲线图 (curve map)
    print("模型正在预测曲线图...")
    curve_map = model.predict(input_tensor)
    print(f"模型输出的曲线图形状: {curve_map.shape}")

    # 4. 将原始图像和曲线图输入到 enhance 函数
    print("正在应用 enhance 函数...")
    # enhance 函数接收的是张量，所以我们使用 input_tensor
    enhanced_image_tensor = enhance(input_tensor, curve_map)
    
    # 5. 后处理以便显示
    # a. 将结果从 Tensor 转换为 NumPy 数组
    enhanced_image_np = enhanced_image_tensor.numpy()
    # b. 移除批次维度，从 (1, 512, 512, 3) -> (512, 512, 3)
    enhanced_image_squeezed = np.squeeze(enhanced_image_np, axis=0)
    # c. (关键步骤!) 将像素值裁剪到 [0, 1] 范围，以防 matplotlib 报错
    enhanced_image_clipped = np.clip(enhanced_image_squeezed, 0, 1)

    print("图像增强完成。正在生成可视化结果...")

    # 6. 使用 Matplotlib 并排显示结果
    plt.figure(figsize=(12, 6)) # 创建一个宽一点的画布

    # 显示原始图片 (预处理后)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_processed)
    plt.title('Original Image (Preprocessed)')
    plt.axis('off') # 不显示坐标轴

    # 显示增强后的图片
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image_clipped)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout() # 调整布局，使其更美观
    plt.show() # 显示图像

if __name__ == '__main__':
    # 检查权重文件是否存在
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"错误: 找不到模型权重文件 '{MODEL_WEIGHTS_PATH}'。请检查路径是否正确。")
    
    # 检查输入图片是否存在
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"错误: 找不到输入图片 '{INPUT_IMAGE_PATH}'。请修改路径或将图片放入 'test_images' 文件夹。")

    # 1. 构建模型架构
    print("正在构建 Keras 模型...")
    dce_model = Model(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), scale_factor=SCALE_FACTOR)

    # 2. 加载训练好的权重
    print(f"正在从 '{MODEL_WEIGHTS_PATH}' 加载权重...")
    dce_model.load_weights(MODEL_WEIGHTS_PATH)
    print("权重加载成功！")

    # 3. 执行可视化
    visualize(dce_model, INPUT_IMAGE_PATH)