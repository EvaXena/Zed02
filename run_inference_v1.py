from predict.predictor import Predictor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model_file = 'result/hgq_dce_proxy_model.h5'  # 确保这个路径和你的模型文件名一致
    input_dir = 'input/'
    output_dir = 'output_keras_decepp/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    predictor = Predictor(model_path=model_file)

    test_images = [f for f in os.listdir(input_dir) if f.endswith(('.png','.jpg','.jpeg'))]

    if not test_images:
        print("no images found")
    for image_name in test_images:
        input_path = os.path.join(input_dir,image_name)
        output_name = os.path.splitext(image_name)[0] + '_depth.png'
        output_path = os.path.join(output_dir,output_name)

        predictor.predict(input_path,output_path)

    print ("All finished")