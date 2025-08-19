#使用原始版本 不添加tanhlu
#模型返回x_r 无监督训练
#enhance模块在损失函数处触发
import tensorflow as tf 
from tensorflow import keras
from HGQ.layers import HQuantize,HConv2D,HConv2DBatchNorm,HActivation
from keras import layers

#定义损失权重
W_col = 0.5
W_tvA = 20


def Hdp_conv(input_tensor,in_ch,out_ch,name_prefix = 'hdp_bn',beta=3e-6):

    x = HConv2DBatchNorm(filters=in_ch,kernel_size=3,padding='same',strides=1,use_bias=False,groups=in_ch,name = f"{name_prefix}_dw_bn",beta = beta)(input_tensor)
    x = HConv2DBatchNorm(filters=out_ch,kernel_size=1,padding='same',strides=1,use_bias=False,name=f"{name_prefix}_pw_bn",beta=beta)(x)
    x = HActivation('relu',name=f"{name_prefix}_relu",beta=beta)(x)
    return x

def Hdp_conv_tanh(input_tensor,in_ch,out_ch,name_prefix = 'hdp_tanh_bn',beta=3e-6):

    x = HConv2DBatchNorm(filters=in_ch,kernel_size=3,padding='same',strides=1,use_bias=False,groups=in_ch,name = f"{name_prefix}_dw_bn",beta = beta)(input_tensor)
    x = HConv2DBatchNorm(filters=out_ch,kernel_size=1,padding='same',strides=1,use_bias=False,name=f"{name_prefix}_pw_bn",beta=beta)(x)
    x = HActivation('tanh',name=f"{name_prefix}_relu",beta=beta)(x)
    return x

def enhance(raw_image,curve_tensor):
    image_tensor = tf.convert_to_tensor(raw_image,dtype=tf.float32)
    curve_tensor = tf.convert_to_tensor(curve_tensor,dtype=tf.float32)

    enhanced_image = raw_image

    for _ in range(8):
        enhanced_image = enhanced_image + curve_tensor * (tf.square(enhanced_image) - enhanced_image)
    
    return enhanced_image


def Model(beta,scale_factor=12,number_f=32):
    input_shape = (600,600,3)
    model_input = keras.Input(shape = input_shape)

    #将所有操作都使用layers来进行
    concat_layer = layers.Concatenate(axis=-1)

    x = HQuantize(beta=beta)(model_input)
    x = layers.AveragePooling2D(pool_size=(scale_factor,scale_factor),name="Downsample_start")(x)

    x = HActivation('linear', name='quant_adapter', beta=beta)(x)


    x1 = Hdp_conv(x,3,number_f,name_prefix='x1_Hdp',beta=beta)
    x2 = Hdp_conv(x1,number_f,number_f,name_prefix='x2_Hdp',beta=beta)
    x3 = Hdp_conv(x2,number_f,number_f,name_prefix='x3_Hdp',beta=beta)
    x4 = Hdp_conv(x3,number_f,number_f,name_prefix='x4_Hdp',beta=beta)

    concat_1 = concat_layer([x3,x4])
    adapt_1 = HActivation('linear', name='quant_adapter_x5', beta=beta)(concat_1)
    x5 = Hdp_conv(adapt_1,number_f *2,number_f,name_prefix='x5_Hdp',beta=beta)

    concat_2 = concat_layer([x2,x5])
    adapt_2 = HActivation('linear', name='quant_adapter_x6', beta=beta)(concat_2)
    x6 = Hdp_conv(adapt_2,number_f *2,number_f,name_prefix='x6_Hdp',beta=beta)

    concat_3 = concat_layer([x1,x6])
    adapt_3 = HActivation('linear', name='quant_adapter_x7', beta=beta)(concat_3)
    x7 = Hdp_conv_tanh(adapt_3,number_f *2,3,name_prefix='x7_Hdp',beta=beta)

    x_r = layers.UpSampling2D(size=(scale_factor,scale_factor),interpolation='bilinear',name = 'upsample_x_r')(x7)

    # x = HActivation('linear', name='quant_adapter_2', beta=beta)(x)
    model = keras.Model(inputs = model_input,outputs=x_r,name='HGQ_DCE_net')
    return model

def loss_spa(original_image,enhanced_image,kernel_size=5):
    #空间一致性损失
    pool = layers.AveragePooling2D(pool_size=kernel_size,strides=1,padding='same')
    #计算原始图像与增强图像的局部均值
    mean_original = pool(original_image)
    mean_enhanced = pool(enhanced_image)

    #计算原始图像与增强图像的局部梯度
    grad_original = tf.abs(original_image - mean_original)
    grad_enhanced = tf.abs(enhanced_image - mean_enhanced)

    return tf.reduce_mean(tf.square(grad_original - grad_enhanced))


def loss_exp(enhanced_image,E=0.6,patch_size=16):
    #曝光控制损失
    pool = layers.AveragePooling2D(pool_size=patch_size)
    avg_luminance = pool(enhanced_image)
    #计算每个块平均亮度与理想值E的L1距离
    d = tf.abs(avg_luminance - E)
    return tf.reduce_mean(d)

def loss_col(enhanced_image):
    #颜色恒常性损失
    #计算RGB三个通道的平均值
    mean_r = tf.reduce_mean(enhanced_image[:,:,:,0])
    mean_g = tf.reduce_mean(enhanced_image[:,:,:,1])
    mean_b = tf.reduce_mean(enhanced_image[:,:,:,2])

    #计算两个之间L2距离的平方
    d_rg = tf.square(mean_r - mean_g)
    d_rb = tf.square(mean_r - mean_b)
    d_gb = tf.square(mean_g - mean_b)

    return d_rg + d_rb + d_gb

def loss_tvA(curve_map):
    #光照平滑度损失
    return tf.reduce_mean(tf.image.total_variation(curve_map))

@tf.function
def train_step(model,original_images,optimizer):
    with tf.GradientTape() as tape:
        #先进行一次前向传播 得到curve图
        curve_map = model(original_images,training=True)

        #手动调用enhanced 生成增强后的图像
        enhanced_images = enhance(original_images,curve_map)

        #计算每个分步损失
        l_spa = loss_spa(original_images,enhanced_images)
        l_exp = loss_exp(enhanced_images)
        l_col = loss_col(enhanced_images)
        l_tvA = loss_tvA(curve_map)

        #计算总损失
        total_loss = l_spa + l_exp + W_col * l_col + W_tvA * l_tvA

    ###
    gradients = tape.gradient(total_loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    ###
    return total_loss

@tf.function
def val_step(model,original_images):
    curve_map = model(original_images,training = False)
    enhanced_images = enhance(original_images,curve_map)
    
    l_spa = loss_spa(original_images,enhanced_images)
    l_exp = loss_exp(enhanced_images)
    l_col = loss_col(enhanced_images)
    l_tvA = loss_tvA(curve_map)

    total_loss = l_spa + l_exp + W_col * l_col + W_tvA * l_tvA
    return total_loss