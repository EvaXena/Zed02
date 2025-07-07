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
#layer1
def Layer1(input_tensor):
    x = layers.Conv2D(filters=32,kernel_size=3,strides=2,use_bias=False,name = 'layer1_conv0')(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn0')(x)
    x = layers.ReLU(max_value=6,name='layer1_relu0')(x)

    print("\nlayer1  stage 0 finished\n")
    
    x = DepthwiseSeparableConv(x,output_channels=32,kernel_size=3,stride=1,name_prefix='layer1_stage1_dws_0')

    print("\nlayer1 stage1 finished\n")

    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=2,name_prefix='layer1_stage2_ir_0')
    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=1,name_prefix='layer1_stage2_ir_1')
    x = InvertedResidual(x,output_channels=32,expansion=6,kernel_size=3,stride=1,name_prefix='layer1_stage2_ir_2')

    print("\nlayer1 stage2 finished\n")

    return x

# ==============================================================================
# 验证阶段：一个最起码的单元测试
# ==============================================================================
if __name__ == '__main__':
    # 设定一个符合预期的输入形状
    input_shape = (256, 256, 3)

    # 创建一个输入节点
    model_input = keras.Input(shape=input_shape)

    # 调用你的建造函数，传入输入节点，得到输出节点
    layer1_output = Layer1(model_input)

    # 用输入和输出来定义一个“临时”的模型，只为了验证layer1的正确性
    layer1_test_model = keras.Model(inputs=model_input, outputs=layer1_output)

    # 打印摘要，检查输出形状是否符合你的计算。
    # 输入(256, 256, 3) -> 第一次下采样 -> (128, 128, ...) -> 第二次下采样 -> (64, 64, ...)
    # 所以最终输出形状应该是 (None, 64, 64, 32)。去核对它！
    print("\n--- Layer 1 Test Model Summary ---")
    layer1_test_model.summary()

    config = hls4ml.utils.config_from_keras_model(layer1_test_model,granularity='model')

    config['Model']['ReuseFactor'] = 1024

    print(config)