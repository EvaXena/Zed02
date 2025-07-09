import tensorflow as tf
import matplotlib.pyplot as plt
from model.midas_small_v1 import Midas_small
from dataloader.load_nyu import NYU_Dataloader

if __name__ == '__main__':
    print("stage 1")

    dataloader = NYU_Dataloader(path='dataset/nyu_depth_v2_labeled.mat',batch_size=16)

    train_dataset,val_dataset = dataloader.get_datasets()

    print("stage 1 finished")

    print("stage 2")

    model = Midas_small()
    model.summary()

    print("stage 2 finished")

    print("stage 3")

    model.compile(
        optimizer='adam',
        loss = 'mean_squared_error',
        metrics=['mean_absolute_error']
    )

    print("stage 3 finished")

    print("stage 4")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='result/midas_small_best.keras', # 保存文件的路径
        save_best_only=True,             # 只保存最好的，不保存最后的
        monitor='val_loss',              # 监控验证集的损失值
        mode='min',                      # 我们希望损失值越小越好
        verbose=1                        # 在保存时打印提示信息
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
    print("stage 4 finished")


    print("start train")
    EPOCHS = 250
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data = val_dataset,
        callbacks=[checkpoint_callback,tensorboard_callback]
    )
    print("train finished")

    # --- 第八步：分析实验结果 ---
    print("Stage 6: Analyzing experimental results...")
    # (你需要先安装matplotlib: pip install matplotlib)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig('training_analysis.png')
    print("Analysis plot saved as 'training_analysis.png'.")
    plt.show()