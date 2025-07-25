# hgq_midas_small.py

import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================================================================
# 第一步：导入HGQ的“未来科技”
# ==============================================================================
from HGQ.layers import HConv2D, HDepthwiseConv2D, HActivation, HQuantize

# ==============================================================================
# 第二步：对你的所有“自定义模块”，进行“H”化改造
# ==============================================================================

#------------------------------------------------------------------------------------------------------------------
def HInvertedResidual(input_tensor, beta, expansion, output_channels, stride, kernel_size=3, name_prefix='inv_res'):
    inter_channel = input_tensor.shape[-1] * expansion
    # 扩张通道
    x = HConv2D(filters=inter_channel, kernel_size=1, use_bias=False, name=f"{name_prefix}_expand_conv", beta=beta)(input_tensor) # <-- 改变在这里!
    x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x) # <-- BN层保持不变!
    x = HActivation('relu', name=f"{name_prefix}_expand_relu", beta=beta)(x) # <-- 改变在这里!
    # 深度卷积
    x = HDepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, name=f"{name_prefix}_dw_conv", beta=beta)(x) # <-- 改变在这里!
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = HActivation('relu', name=f"{name_prefix}_dw_relu", beta=beta)(x) # <-- 改变在这里!
    # 修改通道数
    x = HConv2D(filters=output_channels, kernel_size=1, use_bias=False, name=f"{name_prefix}_project_conv", beta=beta)(x) # <-- 改变在这里!
    x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)
    # 若可行 则残差连接
    if input_tensor.shape[-1] == output_channels and stride == 1:
        x = layers.Add(name=f"{name_prefix}_add")([input_tensor, x]) # <-- Add层保持不变!
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def HDepthwiseSeparableConv(input_tensor, beta, output_channels, kernel_size, stride, name_prefix='dws_conv'):
    x = HDepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=stride, padding='same', use_bias=False, name=f"{name_prefix}_dwconv", beta=beta)(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = HActivation('relu', name=f"{name_prefix}_relu1", beta=beta)(x)
    x = HConv2D(filters=output_channels, kernel_size=(1, 1), strides=1, padding='same', use_bias=False, name=f"{name_prefix}_pwconv", beta=beta)(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def HResidual(input_tensor, beta, filters, name_prefix):
    shortcut = input_tensor
    x = HActivation('relu', name=f"{name_prefix}_relu0", beta=beta)(input_tensor)
    x = HConv2D(filters=filters, kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_conv0", beta=beta)(x)
    x = HActivation('relu', name=f"{name_prefix}_relu1", beta=beta)(x)
    x = HConv2D(filters=filters, kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_conv1", beta=beta)(x)
    x = layers.Add(name=f"{name_prefix}_add")([shortcut, x])
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def HFusionlayer(low_input, high_input, beta, output_channels, name_prefix):
    low_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name=f"{name_prefix}_upsample")(low_input) # <-- UpSampling保持不变!
    low_aligned = HConv2D(filters=high_input.shape[-1], kernel_size=1, padding='same', use_bias=False, name=f"{name_prefix}_align_conv", beta=beta)(low_upsampled)
    x = layers.Add(name=f"{name_prefix}_fuse_add")([low_aligned, high_input])
    # 把beta“契约”传递给子模块
    x = HResidual(x, beta=beta, filters=x.shape[-1], name_prefix=f"{name_prefix}_resunit0")
    x = HResidual(x, beta=beta, filters=x.shape[-1], name_prefix=f"{name_prefix}_resunit1")
    x = HConv2D(filters=output_channels, kernel_size=1, use_bias=False, name=f"{name_prefix}_out_conv", beta=beta)(x)
    return x
#------------------------------------------------------------------------------------------------------------------

# ==============================================================================
# 第三步：构建最终的、完整的HGQ模型
# ==============================================================================

def HLayer1(input_tensor, beta):
    x = HConv2D(filters=32, kernel_size=3, strides=2, use_bias=False, name='layer1_conv0', beta=beta)(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn0')(x)
    x = HActivation('relu', name='layer1_relu0', beta=beta)(x)
    x = HDepthwiseSeparableConv(x, beta=beta, output_channels=24, kernel_size=3, stride=1, name_prefix='layer1_stage1_dws_0')
    x = HInvertedResidual(x, beta=beta, output_channels=32, expansion=6, kernel_size=3, stride=2, name_prefix='layer1_stage2_ir_0')
    x = HInvertedResidual(x, beta=beta, output_channels=32, expansion=6, kernel_size=3, stride=1, name_prefix='layer1_stage2_ir_1')
    x = HInvertedResidual(x, beta=beta, output_channels=32, expansion=6, kernel_size=3, stride=1, name_prefix='layer1_stage2_ir_2')
    return x

def HLayer2(x, beta):
    x = HInvertedResidual(x, beta=beta, output_channels=48, expansion=6, kernel_size=5, stride=2, name_prefix='layer2_stage0_ir_0')
    x = HInvertedResidual(x, beta=beta, output_channels=48, expansion=6, kernel_size=5, stride=1, name_prefix='layer2_stage0_ir_1')
    x = HInvertedResidual(x, beta=beta, output_channels=48, expansion=6, kernel_size=5, stride=1, name_prefix='layer2_stage0_ir_2')
    return x

def HLayer3(x, beta):
    x = HInvertedResidual(x, beta=beta, output_channels=96, expansion=6, kernel_size=3, stride=2, name_prefix='layer3_stage0_ir_0')
    x = HInvertedResidual(x, beta=beta, output_channels=96, expansion=6, kernel_size=3, stride=1, name_prefix='layer3_stage0_ir_1')
    x = HInvertedResidual(x, beta=beta, output_channels=96, expansion=6, kernel_size=3, stride=1, name_prefix='layer3_stage0_ir_2')
    x = HInvertedResidual(x, beta=beta, output_channels=96, expansion=6, kernel_size=3, stride=1, name_prefix='layer3_stage0_ir_3')
    x = HInvertedResidual(x, beta=beta, output_channels=96, expansion=6, kernel_size=3, stride=1, name_prefix='layer3_stage0_ir_4')
    x = HInvertedResidual(x, beta=beta, output_channels=136, expansion=6, kernel_size=5, stride=1, name_prefix='layer3_stage1_ir_0')
    x = HInvertedResidual(x, beta=beta, output_channels=136, expansion=6, kernel_size=5, stride=1, name_prefix='layer3_stage1_ir_1')
    x = HInvertedResidual(x, beta=beta, output_channels=136, expansion=6, kernel_size=5, stride=1, name_prefix='layer3_stage1_ir_2')
    x = HInvertedResidual(x, beta=beta, output_channels=136, expansion=6, kernel_size=5, stride=1, name_prefix='layer3_stage1_ir_3')
    x = HInvertedResidual(x, beta=beta, output_channels=136, expansion=6, kernel_size=5, stride=1, name_prefix='layer3_stage1_ir_4')
    return x

def HLayer4(x, beta):
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=2, name_prefix='layer4_stage0_ir_0')
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=1, name_prefix='layer4_stage0_ir_1')
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=1, name_prefix='layer4_stage0_ir_2')
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=1, name_prefix='layer4_stage0_ir_3')
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=1, name_prefix='layer4_stage0_ir_4')
    x = HInvertedResidual(x, beta=beta, output_channels=232, expansion=6, kernel_size=5, stride=1, name_prefix='layer4_stage0_ir_5')
    x = HInvertedResidual(x, beta=beta, output_channels=384, expansion=6, kernel_size=3, stride=1, name_prefix='layer4_stage1_ir_0')
    return x

def build_hgq_model(beta):
    input_shape = (256, 256, 3)
    model_input = keras.Input(shape=input_shape)

    # 关键的第一步：设置“量化之门”！
    x = HQuantize(beta=beta)(model_input)
    
    # --- 编码器 ---
    layer1_result = HLayer1(x, beta=beta)
    layer2_result = HLayer2(layer1_result, beta=beta)
    layer3_result = HLayer3(layer2_result, beta=beta)
    layer4_result = HLayer4(layer3_result, beta=beta)

    # --- 解码器预处理 ---
    p4 = HConv2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False, name='scratch_stage0_conv_p4', beta=beta)(layer4_result)
    p3 = HConv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False, name='scratch_stage0_conv_p3', beta=beta)(layer3_result)
    p2 = HConv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False, name='scratch_stage0_conv_p2', beta=beta)(layer2_result)
    p1 = HConv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, name='scratch_stage0_conv_p1', beta=beta)(layer1_result)

    # --- 融合层 ---
    x = HFusionlayer(low_input=p4, high_input=p3, beta=beta, output_channels=256, name_prefix='refinenet3')
    x = HFusionlayer(low_input=x, high_input=p2, beta=beta, output_channels=128, name_prefix='refinenet2')
    x = HFusionlayer(low_input=x, high_input=p1, beta=beta, output_channels=64, name_prefix='refinenet1')

    # --- 输出头 ---
    x = HConv2D(filters=32, kernel_size=3, strides=1, padding='same', name='scratch_stageoutput_conv_0', beta=beta)(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name=f"scratch_stageoutput_upsample_0")(x)
    x = HConv2D(filters=32, kernel_size=3, strides=1, padding='same', name='scratch_stageoutput_conv_1', beta=beta)(x)
    x = HActivation('relu', name='scratch_stageoutput_relu_0', beta=beta)(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name=f"scratch_stageoutput_upsample_1")(x)
    # 最终输出层，同样需要被量化
    x = HConv2D(filters=1, kernel_size=1, strides=1, padding='same', name='scratch_stageoutput_conv_2', beta=beta)(x)

    model = keras.Model(inputs=model_input, outputs=x, name='HGQ_Midas_small')
    return model

# ==============================================================================
# 验证阶段
# ==============================================================================
if __name__ == '__main__':
    # 设定一个beta值来实例化模型
    BETA_VALUE = 3e-6 
    hgq_model = build_hgq_model(beta=BETA_VALUE)
    hgq_model.summary(line_length=150)