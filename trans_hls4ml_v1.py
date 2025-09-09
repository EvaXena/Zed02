import hls4ml
import tensorflow as tf
import os
# import plotting
from tensorflow.keras.models import load_model









#v7 使用reusefactor 64 io_parallel 尝试能否综合
#v8 问题并非OOM 而是内存访问有误 降低reusefactor 看能否综合 reusefactor降低到32 无法通过综合
#v9 使用hls4ml进行综合 而不是手动 同V8 失败

#v10 用回io_stream reuse设置128 同时调整时序 converters中调整 clock-period = 10 以满足时序要求 编译都不通过？ 命令行难以完成工作？

if __name__ == "__main__":
    model = load_model('final_keras_decepp_v7.h5')
    model.compile()
    model.summary()
    config = hls4ml.utils.config_from_keras_model(model,granularity='model',backend = 'Vitis')
    default_rf = 128
    # for layer in config['LayerName'].values():
    #     for attr in layer:
    #         if isinstance(attr,str) and 'ReuseFactor' in attr:
    #             layer['ReuseFactor'] = default_rf

    # config['LayerName']['Downsample_start']['ReuseFactor'] = 4096
    config['Model']['Strategy'] = 'Resource'
    config['Model']['ReuseFactor'] = 128


    print("-----------------------------------")
    print("Configuration")
    print("-----------------------------------")

    # hls4ml.utils.print_dict(config)
    
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,hls_config=config,backend="Vitis",output_dir='hlsmodel/hls4ml_prj_v10',part = 'xcu250-figd2104-2L-e',io_type = 'io_stream',clock_period=10
    )
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model11.jpg')
    hls_model.compile()
    reports = hls_model.build(
        csim=False,
        synth=True,
        vsynth=False
    )