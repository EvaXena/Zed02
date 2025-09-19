#训练剪枝后的模型

#服务器优化版
#使用tensorflow 的c++框架
#自适应cpu gpu

#v3 适用于flsea dataset 使用jpg输入 首先修改dataloader
#   适用于zero_decepp 适用于自监督网络

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 先把烦人的日志关掉
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 指定使用的 GPU
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.prune_decepp_v1 import *
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import matplotlib.pyplot as plt
from tqdm import tqdm
from callbacks_v2 import *
import tensorflow_model_optimization as tfmot



#看稀疏度
#----------------------------------------------------------------------------------------------------
def get_model_sparsity(pruned_model):
    """
    通过检查非空的 'pruning_vars' 属性和处理数据类型来计算模型稀疏度。
    这是最稳健的方法，处理了所有边界情况。
    """
    total_params = 0
    total_zeros = 0

    for layer_wrapper in pruned_model.layers:
        if hasattr(layer_wrapper, 'pruning_vars'):
            if layer_wrapper.pruning_vars:
                underlying_layer = layer_wrapper.layer
                for weight in underlying_layer.weights:
                    if 'kernel' in weight.name:
                        mask = layer_wrapper.pruning_vars[0][1]
                        
                        params_in_mask = tf.cast(tf.size(mask), tf.float32)
                        
                        # 【核心修正】将 count_nonzero 的整数结果转换为 float32
                        non_zeros_in_mask = tf.cast(tf.math.count_nonzero(mask), tf.float32)
                        zeros_in_mask = params_in_mask - non_zeros_in_mask
                        
                        total_params += params_in_mask
                        total_zeros += zeros_in_mask
                        
                        break 

    if total_params == 0:
        print("警告: 在模型中没有找到可剪枝的 'kernel'。")
        return tf.constant(0.0)
        
    sparsity = total_zeros / total_params
    return sparsity

class SparsityLogger(Callback):
    """
    一个在每个 epoch 结束后打印并记录模型稀疏度的回调。
    """
    def on_epoch_end(self, epoch, logs=None):
        # 调用辅助函数计算稀疏度
        current_sparsity = get_model_sparsity(self.model)
        
        # 打印到控制台
        print(f" - Sparsity: {current_sparsity.numpy():.2%}")
        
        # 关键：将稀疏度添加到 logs 字典中
        # 这样 CSVLogger 和 TensorBoard 就能自动记录它
        if logs is not None:
            logs['sparsity'] = current_sparsity
#----------------------------------------------------------------------------------------------------




# 图像预处理函数 （flsea版本）
INPUT_SIZE = 600
def load_and_preprocess_image(path):
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img_resized = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img_processed = tf.cast(img_resized, tf.float32) / 255.0
    return img_processed

if __name__ == '__main__':

    # --- 超参数和路径配置 ---
    LR = 1e-5
    EPOCHS = 800
    BATCH_SIZE = 8 # 64 可能会导致显存不足，8 是一个更安全的值
    VAL_SPLIT = 0.15
    DATA_DIR = '/data2/home/clp/Documents/0709_Zed02/0709_Zed02_finished/Zed02/flsea_dataset_jpg'
    patience = 200
    os.makedirs('result', exist_ok=True)
    os.makedirs('reslut/prune_model',exist_ok=True)
    final_model_path = "result/prune_model/prune_keras_decepp_v1.h5"


    
    #stage1 数据加载与处理
    print('Stage 1: Loading dataset paths')
    patterns = [os.path.join(DATA_DIR, '**/*.jpg'), os.path.join(DATA_DIR, '**/*.JPG'),os.path.join(DATA_DIR,'**/*.tiff')]
    all_file_paths = []
    # 使用 glob, 因为我们之前确认它比 list_files 在您的环境中更可靠
    import glob 
    for pattern in patterns:
        all_file_paths.extend(glob.glob(pattern, recursive=True))
    
    if not all_file_paths:
        raise ValueError(f"No images found in {DATA_DIR}. Please check the path.")
    
    np.random.shuffle(all_file_paths)
    
    val_size = int(len(all_file_paths) * VAL_SPLIT)
    train_paths = all_file_paths[val_size:]
    val_paths = all_file_paths[:val_size]
    print(f"Dataset created: {len(train_paths)} training images, {len(val_paths)} validation images")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    # --- 后续的模型创建、编译、训练代码 ---
    print("stage 2: Model setup")

    #导入模型 此处模型在训练内再编译
    #设置优化器
    model_name = 'decepp_prune'
    model = decepp_prune()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    #导入callback 先引入最基础的剪枝callback看看效果
    pruning_callback = pruning_callbacks.UpdatePruningStep()
    pruning_callback.set_model(model)

    #编写自定义训练循环

    
    pruning_callback.on_train_begin()#训练开始 调用callback

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')#此处metrics的本质是什么
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    history = {'loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        print(f"Epoch:{epoch+1}/{EPOCHS}")
        pruning_callback.on_epoch_begin(epoch)#epoch开始 调用callback

        #再一个循环  调用train_dataset
        for step,original_images in enumerate(tqdm(train_dataset,desc="Training")):
            pruning_callback.on_train_batch_begin(step)#batch开始 调用callback
            loss = train_step(model,original_images,optimizer)
            pruning_callback.on_train_batch_end(step,logs={'loss':loss})#batch结束 调用callback

        #一个epoch只调用一次val_loss
        total_val_loss = 0
        num_val_batches = 0
        for original_images in tqdm(val_dataset, desc="Validating"):
            val_loss = val_step(model, original_images)
            total_val_loss += val_loss
            num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")

        #查看稀疏度
        # 调用这个全新的、稳健的函数
        current_sparsity = get_model_sparsity(model)
        
        # 这个打印语句现在可以正常工作了
        print(f" - 模型稀疏度: {current_sparsity.numpy():.2%}")

        pruning_callback.on_epoch_end(epoch, logs={'val_loss': avg_val_loss})#epoch结束 调用callback

    pruning_callback.on_train_end()#训练结束 调用callback
    print("Training finished.")

    # 剥离剪枝层，得到一个干净的、可用于部署的模型
    final_model = strip_pruning(model)
    final_model.save(final_model_path)
    print(f"Final stripped model saved to: {final_model_path}")


        










