#定义全部的回调函数
import tensorflow as tf
import os
import tensorflow_model_optimization as tfmot

logs = 'logs'  # 日志目录

def callbacks_pruning(log_dir=logs,stop_patience = 300,lr_patience = 20,lr_factor = 0.5):
    #stop_patience: 停止训练的耐心值
    #lr_patience: 学习率调整的耐心值
    #lr_factor: 学习率调整的因子
    #返回一个包含所有回调函数的列表
    model_name = "midas_prune_v4"
    steps_per_epoch = 20

    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
        print(f"Created log directory: {log_dir}")

    #1.剪枝回调模块
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    #2.最佳模型存档模块
    #注意保存的模型带有外骨骼
    #如果需要保存去除外骨骼的模型，请在训练结束后使用
    #strip_pruning(model)函数
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('result', f"{model_name}_best_weights.h5"), # 保存文件的路径
        save_best_only=True,             # 只保存最好的，不保存最后的
        save_weights_only=True,  # 只保存权重
        monitor='val_loss',              # 监控验证集的损失值
        mode='min',                      # 我们希望损失值越小越好
        verbose=1                        # 在保存时打印提示信息
    )

    #2.1 最后模型保存模块
    last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('result', f"{model_name}_last_weights.h5"), # 保存文件的路径
        save_weights_only=True,  # 只保存权重
        save_freq=200,       # 每200个batch保存一次
        verbose=1                # 在保存时打印提示信息
    )

    #2.2 定期模型保存模块
    periodic_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('result', f"{model_name}_epoch_{{epoch:03d}}_weights.h5"),
        save_best_only=False,
        save_weights_only=True,
        save_freq=steps_per_epoch * 50, # 关键！我们在这里设定了“每50个epoch”的频率！
        verbose=1
    )


    #3.学习率自动调度模块
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证集的损失值
        factor=lr_factor,    # 学习率调整因子
        patience=lr_patience,  # 学习率调整的耐心值
        min_lr=1e-6,         # 最小学习率
        verbose=1            # 在调整学习率时打印提示信息
    )

    #4.实验终止模块
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 监控验证集的损失值
        patience=stop_patience,  # 停止训练的耐心值
        mode='min',  # 我们希望损失值越小越好
        verbose=1,  # 在停止训练时打印提示信息
        restore_best_weights=False  # 避免剪枝后的权重被覆盖
    )

    #5.实验数据记录模块
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(log_dir, 'training_log.csv'),  # 保存日志的文件路径
        append=True  # 追加到现有文件
        
    )

    #6.数据可视化模块
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,  # 日志目录
        histogram_freq=0,  # 每隔多少个epoch计算直方图
        write_graph=False,  # 是否写入图结构
        write_images=False,  # 是否写入模型权重的图像
        update_freq=50  # 更新频率
    )

    callback_list = [
        pruning_callback,
        best_checkpoint_callback,
        last_checkpoint_callback,
        periodic_checkpoint,
        reduce_lr_callback,
        early_stopping_callback,
        csv_logger_callback,
        tensorboard_callback
    ]

    print("All callbacks are ready.")

    return callback_list
