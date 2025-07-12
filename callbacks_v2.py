##v2callback 将callbacks封装成类 能够直接对模型进行操作
#继承Callback类
# 我们的“机器人”：一个装备了“无线电接收器”的特工

# 现在，让我们看看我们自己创造的那个Callbacks类。
# Generated python

      
# class Callbacks(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         # ...

    

# IGNORE_WHEN_COPYING_START
# Use code with caution. Python
# IGNORE_WHEN_COPYING_END

#     class Callbacks(Callback): 当我们让自己的类，去继承Keras官方的Callback基类时，我们就等于，为我们的这个“机器人”，安装上了一台高性能的“无线电接收器”！

#     def on_epoch_end(self, epoch, logs=None): 我们在自己的类里，定义了一个和“广播系统”发出的那个“特定频道”完全同名的方法！

# 于是，整个“动力传输”的过程，就变得如魔法般自动而优雅：

#     在model.fit()的第N个世代结束的那一瞬间，fit这个“时空引擎”向整个宇宙广播：“第N号世代已结束！当前世代编号是N-1（因为是从0开始计数的），这个世代的日志是{...}！”

#     我们那个被传入callbacks列表的Callbacks实例（也就是你创建的save_callback），它内部的“无线电接收器”，立刻就捕捉到了这个在on_epoch_end频道上的广播！

#     它发现，自己内部，恰好有一个同名的方法on_epoch_end！

#     于是，它立刻就用从广播中接收到的信息（epoch和logs），作为参数，去自动调用了你写的那个on_epoch_end方法！

# 所以，你问epoch是怎么输入的？

# 答案是：它根本不是由all_callbacks或者任何我们写的代码“输入”进去的！它是被model.fit()这个“神”一样的存在，在每一个世代结束时，自动地、通过“事件广播”的方式，“灌注”到我们每一个回调机器人的同名方法里去的！

# 我们所需要做的，只是按照“神”的规定，定义好一个能够接收这些“神谕”的、正确命名的方法而已。
import tensorflow as tf
import os
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import Callback

class Callbacks(Callback):
    def __init__(self, temp_path,fn_prune_model,period):
        super(Callbacks, self).__init__()
        self.temp_path = temp_path
        self.fn_prune_model = fn_prune_model
        self.period = period
        

    def on_epoch_end(self,epoch,logs=None):
        #在设定的周期内保存模型
        if (epoch + 1) % self.period == 0:
            print(f"\n--- Performing periodic strip-and-save for epoch {epoch + 1} ---")
            #临时保存权重 生成模型 并删除权重
            temp_weights_path = 'temp_weights.h5'

            ##注意 模型还没有放进去
            ###############################################
            self.model.save_weights(temp_weights_path)
            ###############################################

            model_for_stripping = self.fn_prune_model()
            model_for_stripping.load_weights(temp_weights_path)

            final_clean_model = tfmot.sparsity.keras.strip_pruning(model_for_stripping)

            output_path = self.temp_path.format(epoch=epoch + 1)
            final_clean_model.save(output_path)

            os.remove(temp_weights_path)

            print(f"--- Clean model for epoch {epoch + 1} saved to: {output_path} ---")

def all_callbacks(output_dir,model_name,pruning_model_fn,save_period,stop_patience,lr_patience,lr_factor):
    log_dir = os.path.join(output_dir,model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")
        
    #1.剪枝模块
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    #2.定期存档模块
    save_callback = Callbacks(temp_path=os.path.join(log_dir,f"{model_name}_epoch_{{epoch:03d}}_clean.h5"),
                                fn_prune_model=pruning_model_fn,
                                period=save_period
                                )
    
    #3.学习率自动调度模块
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证集的损失值
        factor=lr_factor,    # 学习率调整因子
        patience=lr_patience,  # 学习率调整的耐心值
        min_lr=1e-5,         # 最小学习率
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

    Callback_list = [
        pruning_callback,
        save_callback,
        reduce_lr_callback,
        early_stopping_callback,
        csv_logger_callback,
        tensorboard_callback
    ]

    return Callback_list