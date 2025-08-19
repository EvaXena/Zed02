#尝试 将relu int 设置成(3,5)
#同一个深度卷积模块使用相同的relu_int
#在同一个模块内 使用相同的int 保证模块内部数值尺度的一致性

#v3更新 避免一刀切 调整模型不同模块的int
import tensorflow as tf
from tensorflow import keras
from keras import layers
import hls4ml
import qkeras
from qkeras import (
    QConv2D,
    QDepthwiseConv2D,
    QActivation,
    QBatchNormalization,
    quantized_bits,
    quantized_relu
)

import keras_tuner as kt


import numpy as np


#采取分段搜索策略 现将所有的integer都设置为0
# 这样可以避免搜索空间过大导致的性能问题
# 但仍然可以探索不同的bits配置
# 未来可以考虑将integer作为一个超参数进行搜索
# 但这需要更复杂的搜索空间设计
# 目前的设计是：每个卷积层的量化器都可以独立配置
# 但所有的卷积层都共享同一个integer范围
# 这样可以减少搜索空间的复杂度
# 例如：kq1c0_bits = hp.Choice("kq1c0_bits", [4, 6, 8])
#       kq1c0_int = hp.Int("kq1c0_int", 0, 2)
#       kq1c0_quantizer = quantized_bits(bits=kq1c0_bits, integer=kq1c0_int, alpha=1)
# 这样可以确保每个卷积层都可以独立配置bits，但integer保持一致


#------------------------------------------------------------------------------------------------------------------
def QInvertedResidual(input_tensor, hp, expansion, output_channels, stride, kernel_size=3, name_prefix='inv_res'):
    """HP-Ready Version of InvertedResidual."""
    inter_channels = input_tensor.shape[-1] * expansion

    # --- 扩张阶段 (Expansion Stage) ---
    expand_bits = hp.Choice(f"{name_prefix}_exp_bits", [4, 6, 8])
    expand_int = hp.Int(f"{name_prefix}_exp_int", 3, 5)
    expand_quantizer = quantized_bits(bits=expand_bits, integer=expand_int, alpha=1)

    x = QConv2D(
        filters=inter_channels, kernel_size=1, padding='same', use_bias=False,
        name=f"{name_prefix}_expand_conv",
        kernel_quantizer=expand_quantizer
    )(input_tensor)
    x = QBatchNormalization(name=f"{name_prefix}_expand_bn")(x)
    expand_relu_int = hp.Int(f"{name_prefix}_exp_relu_int", 3, 5)
    x = QActivation(activation=quantized_relu(bits=hp.Choice(f"{name_prefix}_exp_act_bits", [6, 8]),integer=expand_relu_int), name=f"{name_prefix}_expand_relu")(x)

    # --- 深度卷积阶段 (Depthwise Stage) ---
    dw_bits = hp.Choice(f"{name_prefix}_dw_bits", [4, 6]) # 深度卷积可以更激进一些
    dw_quantizer = quantized_bits(bits=dw_bits, integer=0, alpha=1) # 深度卷积的权重通常较小

    x = QDepthwiseConv2D(
        kernel_size=kernel_size, strides=stride, padding='same', use_bias=False,
        name=f"{name_prefix}_dw_conv",
        depthwise_quantizer=dw_quantizer # 注意，键名是 depthwise_quantizer
    )(x)
    x = QBatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = QActivation(activation=quantized_relu(bits=hp.Choice(f"{name_prefix}_dw_act_bits", [6, 8]),integer=expand_relu_int), name=f"{name_prefix}_dw_relu")(x)

    # --- 投影阶段 (Projection Stage) ---
    project_bits = hp.Choice(f"{name_prefix}_proj_bits", [6, 8])
    project_int = hp.Int(f"{name_prefix}_proj_int", 0, 2)
    project_quantizer = quantized_bits(bits=project_bits, integer=project_int, alpha=1)
    
    x = QConv2D(
        filters=output_channels, kernel_size=1, padding='same', use_bias=False,
        name=f"{name_prefix}_project_conv",
        kernel_quantizer=project_quantizer
    )(x)
    x = QBatchNormalization(name=f"{name_prefix}_project_bn")(x)

    # 若可行，则添加残差连接
    if input_tensor.shape[-1] == output_channels and stride == 1:
        x = layers.Add(name=f"{name_prefix}_add")([x, input_tensor])
        add_act_bits = hp.Choice(f"{name_prefix}_add_act_bits", [6, 8])
        add_act_int = hp.Int(f"{name_prefix}_add_act_int", 3, 5) # 这个范围也应该由数据驱动
        
        x = QActivation(
            qkeras.quantized_bits(bits=add_act_bits, integer=add_act_int),
            name=f"{name_prefix}_add_requantize"
        )(x)
    return x
#------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------------
def QDepthwiseSeparableConv2D(input_tensor, hp, output_channels, stride, kernel_size=3, name_prefix='dw_conv'):
    """HP-Ready Version of DepthwiseSeparableConv."""
    # --- 深度卷积阶段 (Depthwise Stage) ---
    dw_bits = hp.Choice(f"{name_prefix}_dw_bits", [4, 6])
    dw_int = hp.Int(f"{name_prefix}_dw_int", 0, 2)
    dw_quantizer = quantized_bits(bits=dw_bits, integer=0, alpha=1)
    
    x = QDepthwiseConv2D(
        kernel_size=(kernel_size,kernel_size), strides=stride, padding='same', use_bias=False,
        name=f"{name_prefix}_dw_conv",
        depthwise_quantizer=dw_quantizer
    )(input_tensor)
    x = QBatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = QActivation(activation=quantized_relu(bits=hp.Choice(f"{name_prefix}_act1_bits", [6, 8]),integer=dw_int), name=f"{name_prefix}_relu1")(x)

    # --- 逐点卷积阶段 (Pointwise Stage) ---
    pw_bits = hp.Choice(f"{name_prefix}_pw_bits", [6, 8])
    pw_int = hp.Int(f"{name_prefix}_pw_int", 0, 2)
    pw_quantizer = quantized_bits(bits=pw_bits, integer=pw_int, alpha=1)

    x = QConv2D(
        filters=output_channels, kernel_size=(1,1), strides=1, padding='same', use_bias=False,
        name=f"{name_prefix}_pwconv",
        kernel_quantizer=pw_quantizer
    )(x)
    x = QBatchNormalization(name=f"{name_prefix}_bn2")(x)
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def QResidual(input_tensor, hp, filters, name_prefix='res'):
    """HP-Ready Version of Residual."""
    shortcut = input_tensor

    # 定义这一对卷积层的量化器
    # 可以在一个模块内共享，也可以分开定义，这里我们分开
    conv0_bits = hp.Choice(f"{name_prefix}_c0_bits", [6, 8])
    conv1_bits = hp.Choice(f"{name_prefix}_c1_bits", [6, 8])

    expand_int_0 = hp.Int(f"{name_prefix}_exp_int_0", 0, 2)
    expand_int_1 = hp.Int(f"{name_prefix}_exp_int_1", 0, 2)

    relu_int = hp.Int(f"{name_prefix}_relu_int", 3, 5)
    x = QActivation(activation=quantized_relu(bits=6, integer=relu_int), name=f"{name_prefix}_relu0")(input_tensor)
    x = QConv2D(
        filters=filters, kernel_size=3, padding='same', use_bias=False,
        name=f"{name_prefix}_conv0",
        kernel_quantizer=quantized_bits(bits=conv0_bits, integer=expand_int_0) # 给个固定的integer
    )(x)
    x = QActivation(activation=quantized_relu(bits=6, integer=relu_int), name=f"{name_prefix}_relu1")(x)
    x = QConv2D(
        filters=filters, kernel_size=3, padding='same', use_bias=False,
        name=f"{name_prefix}_conv1",
        kernel_quantizer=quantized_bits(bits=conv1_bits, integer=expand_int_1)
    )(x)

    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def QFusionlayer(low_input, high_input, hp, output_channels, name_prefix='fusion'):
    """HP-Ready Version of Fusionlayer."""
    low_upsample = layers.UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name_prefix}_upsample")(low_input)
    
    expand_int_0 = hp.Int(f"{name_prefix}_exp_int_0", 0, 2)
    expand_int_1 = hp.Int(f"{name_prefix}_exp_int_0", 0, 2)

    # 对齐卷积层 (Align Conv)
    align_bits = hp.Choice(f"{name_prefix}_align_bits", [6, 8])
    align_quantizer = quantized_bits(bits=align_bits, integer=expand_int_0)
    low_aligned = QConv2D(
        filters=high_input.shape[-1], kernel_size=1, padding='same', use_bias=False,
        name=f"{name_prefix}_align_conv",
        kernel_quantizer=align_quantizer
    )(low_upsample)
    
    x = layers.Add(name=f"{name_prefix}_fuse_add")([low_aligned, high_input])

    # 把hp“圣火”传递给子模块！
    x = QResidual(x, hp, filters=x.shape[-1], name_prefix=f"{name_prefix}_resunit0")
    x = QResidual(x, hp, filters=x.shape[-1], name_prefix=f"{name_prefix}_resunit1")

    # 输出卷积层 (Out Conv) - 通常需要更高精度
    out_bits = hp.Choice(f"{name_prefix}_out_bits", [8, 10])
    out_quantizer = quantized_bits(bits=out_bits, integer=expand_int_1)
    x = QConv2D(
        filters=output_channels, kernel_size=1, use_bias=False,
        name=f"{name_prefix}_out_conv",
        kernel_quantizer=out_quantizer
    )(x)
    return x
#------------------------------------------------------------------------------------------------------------------

#正确地调用keras_tuner 不需要构建搜索空间了
#仍然需要写配置文件
#书写规范：kq1c0  kernel_layer1_conv_0

##第一步 书写build_model(hp) 用于构建模型
#注意 在build_model内 需要
#1.为每一层定义超参数
#2.将定义好的超参数转为量化器
#3.将量化器输入对应层
def build_model(hp):


    input_shape = (256,256,3)
    model_input = keras.Input(shape=input_shape)
    #设置一些通用配置
    integer_2 = hp.Int("int_2",min_value=0, max_value=2, default=1)


#------------------------------------------------------------------------------------------------------------------
    #layer 1 
    #配置文件
    k1c0 = qkeras.quantized_bits(
        bits=hp.Choice(("k1c0_bits"),[6,8,10]),
        integer=0,
        alpha=1
    )

    #构建层    
    x = QConv2D(
        filters=32, kernel_size=3, strides=2,padding='same', use_bias=False,
        name='layer1_conv0',
        kernel_quantizer=k1c0
    )(model_input)
    x = QBatchNormalization(name='layer1_bn0')(x)
    x = QActivation(activation=quantized_relu(bits=hp.Choice(f"layer1_relu0_bits", [6, 8])), name='layer1_relu0')(x)

    x = QDepthwiseSeparableConv2D(input_tensor=x,hp=hp,output_channels=24,kernel_size=3,stride=1,name_prefix="layer1_stage1_dws_0")

    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=32,stride=2,kernel_size=3,name_prefix="layer1_stage2_ir_0")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=32,stride=1,kernel_size=3,name_prefix="layer1_stage2_ir_1")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=32,stride=1,kernel_size=3,name_prefix="layer1_stage2_ir_2")
    x1 = x
#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
    #layer 2
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=48,stride=2,kernel_size=5,name_prefix="layer2_stage0_ir_0")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=48,stride=1,kernel_size=5,name_prefix="layer2_stage0_ir_1")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=48,stride=1,kernel_size=5,name_prefix="layer2_stage0_ir_2")
    x2 = x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
    #layer 3
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=96,stride=2,kernel_size=3,name_prefix="layer3_stage0_ir_0")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=96,stride=1,kernel_size=3,name_prefix="layer3_stage0_ir_1")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=96,stride=1,kernel_size=3,name_prefix="layer3_stage0_ir_2")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=96,stride=1,kernel_size=3,name_prefix="layer3_stage0_ir_3")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=96,stride=1,kernel_size=3,name_prefix="layer3_stage0_ir_4")

    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=136,stride=1,kernel_size=5,name_prefix="layer3_stage1_ir_0")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=136,stride=1,kernel_size=5,name_prefix="layer3_stage1_ir_1")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=136,stride=1,kernel_size=5,name_prefix="layer3_stage1_ir_2")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=136,stride=1,kernel_size=5,name_prefix="layer3_stage1_ir_3")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=136,stride=1,kernel_size=5,name_prefix="layer3_stage1_ir_4")
    x3 = x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
    #layer 4
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=2,kernel_size=5,name_prefix="layer4_stage0_ir_0")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=1,kernel_size=5,name_prefix="layer4_stage0_ir_1")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=1,kernel_size=5,name_prefix="layer4_stage0_ir_2")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=1,kernel_size=5,name_prefix="layer4_stage0_ir_3")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=1,kernel_size=5,name_prefix="layer4_stage0_ir_4")
    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=232,stride=1,kernel_size=5,name_prefix="layer4_stage0_ir_5")

    x = QInvertedResidual(input_tensor=x,hp=hp,expansion=6,output_channels=384,stride=1,kernel_size=3,name_prefix="layer4_stage1_ir_0")
    x4 = x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
    #解码器
#------------------------------------------------------------------------------------------------------------------
    #预处理部分 配置文件
    #给不同的layer输出结果配置不同的
    #integer_2 = hp.Int("int_2",min_value=0, max_value=2, default=1)
    layer1_int = hp.Int("layer1_int",min_value=0,max_value=2,default=1)
    layer2_int = hp.Int("layer2_int",min_value=1,max_value=3,default=2)
    layer3_int = hp.Int("layer3_int",min_value=2,max_value=4,default=2)
    layer4_int = hp.Int("layer4_int",min_value=2,max_value=5,default=1)


    kernel_layer1_out_0 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_layer1_out_0_bits"),[4,6,8]),
        integer=layer1_int,
        alpha=1
    )

    kernel_layer2_out_0 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_layer2_out_0_bits"),[4,6,8]),
        integer=layer2_int,
        alpha=1
    )

    kernel_layer3_out_0 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_layer3_out_0_bits"),[4,6,8]),
        integer=layer3_int,
        alpha=1
    )

    kernel_layer4_out_0 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_layer4_out_0_bits"),[4,6,8]),
        integer=layer4_int,
        alpha=1
    )

    #卷积预处理
    p4 = QConv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False,
                 name='layer4_out_conv0',
                 kernel_quantizer=kernel_layer4_out_0)(x4)
    
    p3 = QConv2D(filters=256,kernel_size=3,strides=1,padding='same',use_bias=False,
                 name='layer3_out_conv0',
                 kernel_quantizer=kernel_layer3_out_0)(x3)
    
    p2 = QConv2D(filters=128,kernel_size=3,strides=1,padding='same',use_bias=False,
                 name='layer2_out_conv0',
                 kernel_quantizer=kernel_layer2_out_0)(x2)
    
    p1 = QConv2D(filters=64,kernel_size=3,strides=1,padding='same',use_bias=False,
                 name='layer1_out_conv0',
                 kernel_quantizer=kernel_layer1_out_0)(x1)
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
    #融合层

    x = QFusionlayer(low_input=p4, high_input=p3, hp=hp, output_channels=256, name_prefix='fusion_4_3')
    x = QFusionlayer(low_input=x, high_input=p2, hp=hp, output_channels=128, name_prefix='fusion_3_2')
    x = QFusionlayer(low_input=x, high_input=p1, hp=hp, output_channels=64, name_prefix='fusion_2_1')
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
    #输出层
#------------------------------------------------------------------------------------------------------------------
    #输出层配置文件
    int_final_out_0 = hp.Int("int_final_out_0",min_value=0,max_value=4,default=1)
    int_final_out_1 = hp.Int("int_final_out_1",min_value=0,max_value=4,default=1)
    int_final_out_2 = hp.Int("int_final_out_2",min_value=0,max_value=4,default=1)

    kernel_final_out_0 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_final_out_0_bits"),[4,6,8]),
        integer=int_final_out_0,
        alpha=1
    )

    kernel_final_out_1 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_final_out_1_bits"),[4,6,8]),
        integer=int_final_out_1,
        alpha=1
    )

    kernel_final_out_2 = qkeras.quantized_bits(
        bits=hp.Choice(("kernel_final_out_2_bits"),[6,8,10]),
        integer=int_final_out_2,
        alpha=1
    )

    

    #输出层
    x = QConv2D(filters=32,kernel_size=3,strides=1,padding='same',use_bias=False,name='final_out_conv0',
                 kernel_quantizer=kernel_final_out_0)(x)
    x = layers.UpSampling2D(size=(2,2), interpolation='nearest', name='final_out_upsample_0')(x)
    x = QConv2D(filters=32,kernel_size=3,strides=1,padding='same',use_bias=False,name='final_out_conv1',
                 kernel_quantizer=kernel_final_out_1)(x)
    x = QActivation(activation=quantized_relu(bits=hp.Choice("final_out_relu_bits", [6, 8])), name='final_out_relu_0')(x)
    x = layers.UpSampling2D(size=(2,2), interpolation='nearest', name='final_out_upsample_1')(x)
    x = QConv2D(filters=1,kernel_size=1,strides=1,padding='same',use_bias=False,name='final_out_conv2',
                 kernel_quantizer=kernel_final_out_2)(x)

#------------------------------------------------------------------------------------------------------------------
    #输出模型
    model = keras.Model(inputs=model_input, outputs=x, name='Q_MiDaS_Small_v2')

    #设置模型优化器与参数
    lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling='LOG', default=1e-4)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae']
    )


    return model


