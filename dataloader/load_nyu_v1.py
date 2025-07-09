import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class NYU_Dataloader:
    def __init__(self,path,batch_size=16,val_split=0.15,random_state=42):
        self.path = path
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.max_depth = None

        self._load_and_preprocess()
        self._create_datasets()

    def _load_and_preprocess(self):
        print("Initiating data loading from HDF5 file...")
        with h5py.File(self.path,'r') as file:
            images = file['images'][:]
            depths = file['depths'][:]

        
        print("Transposing")
        images = np.transpose(images,(0,2,3,1))

        print("Normalizing")

        images = images.astype('float32') / 255.0
        depths = depths.astype('float32') 

        self.max_depth = np.max(depths)

        depths /= self.max_depth

        self.images = images
        self.depths = depths

        print(f"Max depth is :{self.max_depth}")

    def _create_datasets(self):
        #直接打包numpy数据
        X_train,X_val,Y_train,Y_val = train_test_split(self.images,self.depths,test_size=self.val_split,random_state=self.random_state)

        Y_train = np.expand_dims(Y_train,axis=-1)
        Y_val = np.expand_dims(Y_val,axis=-1)

        self.train_dataset = self._build_pipeline(X_train,Y_train,shuffle=True)
        self.val_dataset = self._build_pipeline(X_val,Y_val,shuffle=False)

        print("Data pipelines are ready")


    def _build_pipeline(self,images,depths,shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((images,depths))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def get_datasets(self):
        return self.train_dataset,self.val_dataset
    
    def get_normalization_params(self):
        return {'max_depth':self.max_depth}










