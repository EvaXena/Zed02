# =================================================================
#             纯 Keras 版本的 Zero-DCE++ 模型
# =================================================================
# - 已移除所有 HGQ 特有层和参数 (HQuantize, HConv2DBatchNorm, HActivation, beta)
# - 使用标准的 Keras 层 (DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, etc.)
# - 核心网络架构、损失函数和训练逻辑保持不变

#cat使用layers层
#插值使用最临近
#分开concat层 避免dataflow难以识别
#损失函数疑似有误 进行确认 修改了W_col 修改了l_col
import tensorflow as tf 
from tensorflow import keras
from keras import layers

# --- 常量定义 (保持不变) ---
W_col = 0.5
W_tvA = 20.0

# --- 标准 Keras 版本的深度可分离卷积块 ---
def dsc_block(input_tensor, filters, name_prefix):
    """
    一个标准的深度可分离卷积块，等效于原来的 Hdp_conv。
    包含: DepthwiseConv -> BatchNorm -> ReLU -> PointwiseConv -> BatchNorm -> ReLU
    """
    # 深度卷积部分
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_dw")(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    
    # 逐点卷积部分
    x = layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu")(x)
    return x

def dsc_block_tanh(input_tensor, filters, name_prefix):
    """
    与上面相同，但最后使用 tanh 激活函数，等效于原来的 Hdp_conv_tanh。
    """
    # 深度卷积部分
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_dw")(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    
    # 逐点卷积部分
    x = layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.Activation('tanh', name=f"{name_prefix}_tanh")(x)
    return x

# --- 增强函数 (保持不变, 它与模型层类型无关) ---
def enhance(raw_image, curve_tensor):
    enhanced_image = raw_image
    for _ in range(8):
        enhanced_image = enhanced_image + curve_tensor * (tf.square(enhanced_image) - enhanced_image)
    return enhanced_image

# --- 标准 Keras 模型定义 ---
def Model(input_shape=(600, 600, 3), scale_factor=12, number_f=32):
    # beta 参数已移除
    model_input = keras.Input(shape=input_shape)
    
    # 移除了 HQuantize，直接从输入开始
    x = layers.AveragePooling2D(pool_size=(scale_factor, scale_factor), name="Downsample_start")(model_input)

    # 移除了第一个 HActivation 适配器

    #将所有操作都使用layers来进行
    concat_layer1 = layers.Concatenate(name='Concat_1',axis=-1)
    concat_layer2 = layers.Concatenate(name='Concat_2',axis=-1)
    concat_layer3 = layers.Concatenate(name='Concat_3',axis=-1)


    x1 = dsc_block(x, number_f, name_prefix='x1')
    x2 = dsc_block(x1, number_f, name_prefix='x2')
    x3 = dsc_block(x2, number_f, name_prefix='x3')
    x4 = dsc_block(x3, number_f, name_prefix='x4')

    concat_1 = concat_layer1([x3,x4])
    # 移除了 HActivation 适配器，直接连接
    x5 = dsc_block(concat_1, number_f, name_prefix='x5')

    concat_2 = concat_layer2([x2,x5])
    # 移除了 HActivation 适配器
    x6 = dsc_block(concat_2, number_f, name_prefix='x6')

    concat_3 = concat_layer3([x1,x6])
    # 移除了 HActivation 适配器
    x7 = dsc_block_tanh(concat_3, 3, name_prefix='x7')

    x_r = layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation='nearest', name='upsample_x_r')(x7)

    model = keras.Model(inputs=model_input, outputs=x_r, name='Keras_DCE_Net')
    return model

# --- 损失函数 (保持不变, 它们与模型层类型无关) ---
def loss_spa(original_image, enhanced_image, kernel_size=5):
    pool = layers.AveragePooling2D(pool_size=kernel_size, strides=1, padding='same')
    mean_original = pool(original_image)
    mean_enhanced = pool(enhanced_image)
    grad_original = tf.abs(original_image - mean_original)
    grad_enhanced = tf.abs(enhanced_image - mean_enhanced)
    return tf.reduce_mean(tf.square(grad_original - grad_enhanced))

def loss_exp(enhanced_image, E=0.6, patch_size=16):
    pool = layers.AveragePooling2D(pool_size=patch_size)
    avg_luminance = pool(enhanced_image)
    d = tf.abs(avg_luminance - E)
    return tf.reduce_mean(d)

#修改为单张图片计算损失
def loss_col(enhanced_image):
    mean_rgb = tf.reduce_mean(enhanced_image,axis = [1,2],keepdims=False)

    mean_r = mean_rgb[:,0]
    mean_g = mean_rgb[:,1]
    mean_b = mean_rgb[:,2]

    d_rg = tf.square(mean_r - mean_g)
    d_rb = tf.square(mean_r - mean_b)
    d_gb = tf.square(mean_g - mean_b)

    per_image_loss = d_rg + d_rb + d_gb

    return tf.reduce_mean(per_image_loss)

def loss_tvA(curve_map):
    return tf.reduce_mean(tf.image.total_variation(curve_map))

# --- 训练和验证步骤 (保持不变, 它们与模型对象和张量交互的方式相同) ---
@tf.function
def train_step(model, original_images, optimizer):
    with tf.GradientTape() as tape:
        curve_map = model(original_images, training=True)
        enhanced_images = enhance(original_images, curve_map)
        total_loss = loss_spa(original_images, enhanced_images) + loss_exp(enhanced_images) + W_col * loss_col(enhanced_images) + W_tvA * loss_tvA(curve_map)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

@tf.function
def val_step(model, original_images):
    curve_map = model(original_images, training=False)
    enhanced_images = enhance(original_images, curve_map)
    total_loss = loss_spa(original_images, enhanced_images) + loss_exp(enhanced_images) + W_col * loss_col(enhanced_images) + W_tvA * loss_tvA(curve_map)
    return total_loss