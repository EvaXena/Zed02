import tensorflow as tf
from tensorflow import keras
from keras import layers
import hls4ml

#------------------------------------------------------------------------------------------------------------------
#倒置残差
def InvertedResidual(input_tensor,expansion,output_channels,stride,kernel_size = 3,name_prefix = 'inv_res'):
    inter_channel = input_tensor.shape[-1] * expansion
    #扩张通道
    x = layers.Conv2D(filters=inter_channel,kernel_size=1,use_bias=False,name=f"{name_prefix}_expand_conv")(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x)
    x = layers.ReLU(max_value=6,name=f"{name_prefix}_expand_relu")(x)
    #深度卷积
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding='same',use_bias=False,name=f"{name_prefix}_dw_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.ReLU(max_value=6,name=f"{name_prefix}_dw_relu")(x)
    #修改通道数为output_channels
    x = layers.Conv2D(filters = output_channels,kernel_size=1,use_bias=False,name=f"{name_prefix}_project_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)
    #若可行 则残差连接
    if input_tensor.shape[-1] == output_channels and stride == 1:
        x = layers.Add(name=f"{name_prefix}_add")([input_tensor,x])
    return x
#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
#深度 可分离 卷积
def DepthwiseSeparableConv(input_tensor,output_channels,kernel_size,stride,name_prefix = 'dws_conv'):
    x = layers.DepthwiseConv2D(kernel_size=(kernel_size,kernel_size),strides=stride,padding='same',use_bias=False,
                               name = f"{name_prefix}_dwconv")(input_tensor)
    x = layers.BatchNormalization(name = f"{name_prefix}_bn1")(x)
    x = layers.ReLU(max_value=6,name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(filters=output_channels,kernel_size=(1,1),strides=1,padding='same',use_bias=False,name=f"{name_prefix}_pwconv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    #逐点卷积后没有激活函数
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#残差卷积
def Residual(input_tensor,filters,name_prefix):
    shortcut = input_tensor

    x = layers.ReLU(name=f"{name_prefix}_relu0")(input_tensor)
    x = layers.Conv2D(filters=filters,kernel_size=3,padding='same',use_bias=False,name=f"{name_prefix}_conv0")(x)

    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(filters=filters,kernel_size=3,padding='same',use_bias=False,name=f"{name_prefix}_conv1")(x)

    x = layers.Add(name=f"{name_prefix}_add")([shortcut,x])
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#融合层
def Fusionlayer(low_input,high_input,output_channels,name_prefix):
    low_upsampled = layers.UpSampling2D(size=(2,2),interpolation='nearest',name=f"{name_prefix}_upsample")(low_input)

    low_aligned = layers.Conv2D(filters=high_input.shape[-1],kernel_size=1,padding='same',use_bias=False,name=f"{name_prefix}_align_conv")(low_upsampled)

    x = layers.Add(name=f"{name_prefix}_fuse_add")([low_aligned,high_input])

    x = Residual(x,filters=x.shape[-1],name_prefix=f"{name_prefix}_resunit0")
    x = Residual(x,filters=x.shape[-1],name_prefix=f"{name_prefix}_resunit1")

    x = layers.Conv2D(filters=output_channels,kernel_size=1,use_bias=False,name=f"{name_prefix}_out_conv")(x)
    return x
#------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------
#layer1
def Layer1(input_tensor):
    x = layers.Conv2D(filters=32,kernel_size=3,strides=2,use_bias=False,name = 'layer1_conv0')(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn0')(x)
    x = layers.ReLU(max_value=6,name='layer1_relu0')(x)

    print("\nlayer1  stage 0 finished\n")
    
    x = DepthwiseSeparableConv(x,output_channels=24,kernel_size=3,stride=1,name_prefix='layer1_stage1_dws_0')

    print("\nlayer1 stage1 finished\n")

    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=2,name_prefix='layer1_stage2_ir_0')
    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=1,name_prefix='layer1_stage2_ir_1')
    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=1,name_prefix='layer1_stage2_ir_2')

    print("\nlayer1 stage2 finished\n")

    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#layer2
def Layer2(x):
    x = InvertedResidual(x,output_channels=48,expansion=6,kernel_size=5,stride=2,name_prefix='layer2_stage0_ir_0')
    x = InvertedResidual(x,output_channels=48,expansion=6,kernel_size=5,stride=1,name_prefix='layer2_stage0_ir_1')
    x = InvertedResidual(x,output_channels=48,expansion=6,kernel_size=5,stride=1,name_prefix='layer2_stage0_ir_2')

    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#layer3
def Layer3(x):
    x = InvertedResidual(x,output_channels=96,expansion=6,kernel_size=3,stride=2,name_prefix='layer3_stage0_ir_0')
    x = InvertedResidual(x,output_channels=96,expansion=6,kernel_size=3,stride=1,name_prefix='layer3_stage0_ir_1')
    x = InvertedResidual(x,output_channels=96,expansion=6,kernel_size=3,stride=1,name_prefix='layer3_stage0_ir_2')
    x = InvertedResidual(x,output_channels=96,expansion=6,kernel_size=3,stride=1,name_prefix='layer3_stage0_ir_3')
    x = InvertedResidual(x,output_channels=96,expansion=6,kernel_size=3,stride=1,name_prefix='layer3_stage0_ir_4')

    x = InvertedResidual(x,output_channels=136,expansion=6,kernel_size=5,stride=1,name_prefix='layer3_stage1_ir_0')
    x = InvertedResidual(x,output_channels=136,expansion=6,kernel_size=5,stride=1,name_prefix='layer3_stage1_ir_1')
    x = InvertedResidual(x,output_channels=136,expansion=6,kernel_size=5,stride=1,name_prefix='layer3_stage1_ir_2')
    x = InvertedResidual(x,output_channels=136,expansion=6,kernel_size=5,stride=1,name_prefix='layer3_stage1_ir_3')
    x = InvertedResidual(x,output_channels=136,expansion=6,kernel_size=5,stride=1,name_prefix='layer3_stage1_ir_4')

    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#layer4
def Layer4(x):
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=2,name_prefix='layer4_stage0_ir_0')
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=1,name_prefix='layer4_stage0_ir_1')
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=1,name_prefix='layer4_stage0_ir_2')
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=1,name_prefix='layer4_stage0_ir_3')
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=1,name_prefix='layer4_stage0_ir_4')
    x = InvertedResidual(x,output_channels=232,expansion=6,kernel_size=5,stride=1,name_prefix='layer4_stage0_ir_5')

    x = InvertedResidual(x,output_channels=384,expansion=6,kernel_size=3,stride=1,name_prefix='layer4_stage1_ir_0')
    return x
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#解码器
def Midas_small():

    input_shape = (256,256,3)
    model_input = keras.Input(shape=input_shape)

    #保存编码器的输出
    layer1_result = Layer1(model_input)
    layer2_result = Layer2(layer1_result)
    layer3_result = Layer3(layer2_result)
    layer4_result = Layer4(layer3_result)

    #解码器预处理
    p4 = layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False,name='scratch_stage0_conv_p4')(layer4_result)
    p3 = layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',use_bias=False,name='scratch_stage0_conv_p3')(layer3_result)
    p2 = layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',use_bias=False,name='scratch_stage0_conv_p2')(layer2_result)
    p1 = layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',use_bias=False,name='scratch_stage0_conv_p1')(layer1_result)

    #融合层进行融合
    x = Fusionlayer(low_input=p4,high_input=p3,output_channels=256,name_prefix='refinenet3')
    x = Fusionlayer(low_input=x,high_input=p2,output_channels=128,name_prefix='refinenet2')
    x = Fusionlayer(low_input=x,high_input=p1,output_channels=64,name_prefix='refinenet1')

    #创建输出头
    x = layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same',name='scratch_stageoutput_conv_0')(x)
    x = layers.UpSampling2D(size=(2,2),interpolation='nearest',name=f"scratch_stageoutput_upsample_0")(x)
    x = layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same',name='scratch_stageoutput_conv_1')(x)
    x = layers.ReLU(name='scratch_stageoutput_relu_0')(x)
    x = layers.UpSampling2D(size=(2,2),interpolation='nearest',name=f"scratch_stageoutput_upsample_1")(x)
    x = layers.Conv2D(filters=1,kernel_size=1,strides=1,padding='same',name='scratch_stageoutput_conv_2')(x)
    x = layers.ReLU(name='scratch_stageoutput_relu_1')(x)

    model = keras.Model(inputs=model_input,outputs=x,name='Midas_small')

    return model


# ==============================================================================
# 验证阶段：一个最起码的单元测试
# ==============================================================================
if __name__ == '__main__':
    # 设定一个符合预期的输入形状
    model = Midas_small()

    model.summary(line_length=150)

    keras.utils.plot_model(model,to_file='midas_small.svg',show_shapes=True)

    config = hls4ml.utils.config_from_keras_model(model,granularity='model')

    config['Model']['ReuseFactor'] = 1024

    print(config)