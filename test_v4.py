from predict.predictor import Predictor
import numpy as np

if __name__ == '__main__':
    predictor = Predictor(model_path='result/midas_small_best.h5')
    
    # 加载我们刚刚保存的纯净样本
    pure_sample_path = 'input/control_sample_100.npy'
    pure_numpy_array = np.load(pure_sample_path)
    
    # 调用新的方法！
    depth_map_result = predictor.predict_from_numpy(pure_numpy_array)
    
    if np.mean(depth_map_result) > 0:
        print("\nDIAGNOSIS CONFIRMED: The model is ALIVE!")
        print("The 'all-zero' issue was caused by JPEG compression artifacts.")
        predictor.save_visual_depth_map(depth_map_result, 'output_depth_maps/PURE_TEST_SUCCESS.png')
        print("A non-black output has been saved. The curse is broken.")
    else:
        print("\nIMPOSSIBLE... The paradox continues.")