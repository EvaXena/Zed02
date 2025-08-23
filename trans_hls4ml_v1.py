import hls4ml
import tensorflow as tf
import os

os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
 
from tensorflow.keras.models import load_model

model = load_model('final_keras_decepp_v2.h5')
config = hls4ml.utils.config_from_keras_model(model,granularity='model',backend='Vitis')
config['Model']['ReuseFactor'] = 8192
config['Model']['Strategy'] = 'Resource'
config['Model']['Precision'] = 'ap_fixed<12,4>'
print("-----------------------------------")
print("Configuration")
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,hls_config=config,backend="Vitis",output_dir='hlsmodel/hls4ml_prj_v1',part = 'xcu250-figd2104-2L-e'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model1.svg')
hls_model.compile(csim=False,synth=True)