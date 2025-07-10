#剪枝后的模型内部包含编译，不再run内编译
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.python.core.sparsity.keras import prune,pruning_callbacks,pruning_schedule

def Midas_prune():
    """
    剪枝Midas模型
    """
    # 加载预训练的Midas模型
    print("Loading baseline model...")

    baseline_model = load_model('result/midas_small_best_v2.h5')

    # 定义稀疏化参数
    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.3,  # 30% 稀疏化
            begin_step=20000,  # 从第20000步开始稀疏化
            frequency=1000  # 每1000步更新一次稀疏化
        )
    }

    # 包装剪枝外骨骼
    model_to_prune = prune.prune_low_magnitude(
        baseline_model,
        **pruning_params
    )

    # 编译模型
    adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model_to_prune.compile(
        optimizer=adam,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    model_to_prune.summary()
    return model_to_prune