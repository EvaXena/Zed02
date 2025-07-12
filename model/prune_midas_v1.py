#剪枝后的模型内部包含编译，不再run内编译
#剪枝模型应该不应该续训而是从头开始训练
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.python.core.sparsity.keras import prune,pruning_callbacks,pruning_schedule
from model.midas_small_v2 import Midas_small

def Midas_prune():
    """
    剪枝Midas模型
    """
    # 加载预训练的Midas模型
    print("Loading baseline model...")

    baseline_model = Midas_small()

    # 定义稀疏化参数
    pruning_params = {
        "pruning_schedule": pruning_schedule.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5, # 我们可以从一个更保守的50%稀疏度开始
            begin_step=1000,     # 在第1000步（大约第50个epoch）开始剪枝
            end_step=6000       # 在第6000步（大约第300个epoch）达到目标稀疏度
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