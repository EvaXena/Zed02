import os

if __name__ == "__main__":
    
    img_path = "flsea_train_dataset/imgs"
    pwd = os.getcwd()
    data_path = os.path.join(pwd,img_path)
    if os.path.isdir(data_path):
        imgs = os.listdir(data_path)
        for img in imgs:
            name = os.path.splitext(img)
            new_path = os.path.join(data_path,str(name[0])+'.jpg')
            old_path = os.path.join(data_path,str(name[0])+'.tiff')
            os.rename(old_path,new_path)
            print(f"changed {old_path} to {new_path}")
            # os.path.join()