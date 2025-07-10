#定义全部的回调函数
import tensorflow as tf
import os
import tensorflow_model_optimization as tfmot

logs = 'logs'  # 日志目录

def callbacks_pruning(log_dir=logs,stop_patience = 20,lr_patience = 5,lr_factor = 0.5):
    #stop_patience: 停止训练的耐心值
    #lr_patience: 学习率调整的耐心值
    #lr_factor: 学习率调整的因子
    #返回一个包含所有回调函数的列表

    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
        print(f"Created log directory: {log_dir}")

    #1.剪枝回调模块
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    #2.最佳模型存档模块
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, 'midas_prune_best_v1.h5'), # 保存文件的路径
        save_best_only=True,             # 只保存最好的，不保存最后的
        save_weights_only=False,  # 保存整个模型而不是仅仅权重
        monitor='val_loss',              # 监控验证集的损失值
        mode='min',                      # 我们希望损失值越小越好
        verbose=1                        # 在保存时打印提示信息
    )

    #3.学习率自动调度模块
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证集的损失值
        factor=lr_factor,    # 学习率调整因子
        patience=lr_patience,  # 学习率调整的耐心值
        min_lr=1e-7,         # 最小学习率
        verbose=1            # 在调整学习率时打印提示信息
    )

    #4.实验终止模块
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 监控验证集的损失值
        patience=stop_patience,  # 停止训练的耐心值
        mode='min',  # 我们希望损失值越小越好
        verbose=1,  # 在停止训练时打印提示信息
        restore_best_weights=True  # 恢复最佳权重
    )

    #5.实验数据记录模块
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(log_dir, 'training_log.csv'),  # 保存日志的文件路径
        append=True  # 追加到现有文件
        
    )

    #6.数据可视化模块
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,  # 日志目录
        histogram_freq=1,  # 每隔多少个epoch计算直方图
        write_graph=True,  # 是否写入图结构
        write_images=True,  # 是否写入模型权重的图像
        update_freq='epoch'  # 更新频率
    )

    callback_list = [
        pruning_callback,
        checkpoint_callback,
        reduce_lr_callback,
        early_stopping_callback,
        csv_logger_callback,
        tensorboard_callback
    ]

    print("All callbacks are ready.")

    return callback_list
