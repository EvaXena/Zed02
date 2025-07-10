import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import os

class Comparator:
    def __init__(self,baseline_model_path,compared_model_path):
        """
        初始化比较器，加载基准模型和被比较模型。
        """
        print("--- Initializing Model Comparator ---")
        self.baseline_model = self._load_model(baseline_model_path,"Baseline")
        self.compared_model = self._load_model(compared_model_path,"Compared")
        print("--- Models loaded successfully ---")

        self.baseline_metrics = {}
        self.compared_metrics = {}

        print("--------------------------------------\n")

    def _load_model(self,model_path,model_name):
        print(f"Loading {model_name} model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name} model not found at: {model_path}")
        return load_model(model_path)
    
    def _count_non_zero_weights(self,model):
        non_zero = 0
        total = 0
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                weights = layer.get_weights()
                for w in weights:
                    non_zero += tf.math.count_nonzero(w).numpy()
                    total += tf.size(w).numpy()
        return non_zero, total
    
    def _get_flops(self,model):
        try:
            input_shape = [1] + model.input_shape[1:]
            inputs = [tf.TensorSpec(input_shape, model.inputs[0].dtype)]
            
            forward_pass = tf.function(model.call, input_signature=inputs)
            graph_info = tf.compat.v1.profiler.profile(
                graph=forward_pass.get_concrete_function().graph,
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            )
            return graph_info.total_float_ops // 2
        except Exception:
            # 在某些复杂模型或TF版本上，profiler可能会失败
            return -1
        
    def compare_model_size(self):
        print("--- Round 1: Model Size (Parameter Count) ---")

        baseline_total_params = self.baseline_model.count_params()
        self.baseline_metrics['total_params'] = baseline_total_params

        non_zero,total = self._count_non_zero_weights(self.compared_model)

        self.compared_metrics['total_params'] = total
        self.compared_metrics['non_zero_params'] = non_zero

        sparsity = 1.0 - (non_zero / total) if total > 0 else 0

        self.compared_metrics['sparsity'] = sparsity

        print(f"Compared Model Total Parameters:   {total:,}")
        print(f"Compared Model Non-Zero Parameters: {non_zero:,}")
        print(f"Achieved Sparsity: {sparsity:.2%}")
        print("--------------------------------------------\n")
        return self

    def compare_flops(self):
        """对比并打印模型的理论计算量。"""
        print("--- Round 2: Computational Cost (FLOPs) ---")
        
        flops_baseline = self._get_flops(self.baseline_model)
        self.baseline_metrics['flops'] = flops_baseline
        if flops_baseline != -1:
            print(f"Baseline Model FLOPs (approx.): {flops_baseline / 1e6:.2f} M-FLOPs")
        else:
            print("Could not profile Baseline Model FLOPs.")
            
        flops_compressed = self._get_flops(self.compressed_model)
        if flops_compressed != -1:
            sparsity = self.compressed_metrics.get('sparsity', 0)
            theoretical_flops = flops_compressed * (1 - sparsity)
            self.compressed_metrics['flops_theoretical'] = theoretical_flops
            print(f"Compressed Model FLOPs (theoretical): {theoretical_flops / 1e6:.2f} M-FLOPs")
        else:
            print("Could not profile Compressed Model FLOPs.")
            
        print("Note: True benefit is in reduced DSP usage during HLS synthesis.")
        print("--------------------------------------------\n")
        return self
    
    def compare_accuracy(self, X_test: np.ndarray, y_test: np.ndarray, classes: list, batch_size: int = 1024):
        """
        对比并打印模型的预测精度，并绘制ROC曲线。
        
        参数:
            X_test, y_test: 测试数据集。
            classes (list): 类别名称列表，用于图例。
            batch_size (int): 预测时使用的批次大小。
        """
        print("--- Round 3: Predictive Accuracy ---")
        
        print("Generating predictions...")
        y_baseline = self.baseline_model.predict(X_test, batch_size=batch_size)
        y_compressed = self.compressed_model.predict(X_test, batch_size=batch_size)
        
        acc_baseline = np.mean(tf.keras.metrics.categorical_accuracy(y_test, y_baseline))
        acc_compressed = np.mean(tf.keras.metrics.categorical_accuracy(y_test, y_compressed))
        self.baseline_metrics['accuracy'] = acc_baseline
        self.compressed_metrics['accuracy'] = acc_compressed
        
        accuracy_drop = acc_baseline - acc_compressed
        
        print(f"Baseline Model Accuracy: {acc_baseline:.4f}")
        print(f"Compressed Model Accuracy:   {acc_compressed:.4f}")
        print(f"Accuracy Drop:   {accuracy_drop:.4f} (Drop of {(accuracy_drop/acc_baseline):.2%})")
        
        print("\nPlotting ROC curves for detailed comparison...")
        fig, ax = plt.subplots(figsize=(9, 9))
        plotting.makeRoc(y_test, y_baseline, classes, linestyle='-')
        plt.gca().set_prop_cycle(None)
        plotting.makeRoc(y_test, y_compressed, classes, linestyle='--')
        
        lines = [Line2D([0], [0], ls='-'), Line2D([0], [0], ls='--')]
        leg = Legend(ax, lines, labels=['Baseline Model', 'Compressed Model'], loc='lower right', frameon=False)
        ax.add_artist(leg)
        
        plt.savefig('comparison_report.png')
        print("Comparison ROC curve saved to 'comparison_report.png'")
        plt.show()
        print("--------------------------------------------\n")
        return self
        
    def generate_summary_report(self):
        """打印一份最终的总结报告。"""
        print("========== FINAL COMPARISON REPORT ==========")
        print(f"{'Metric':<25} | {'Baseline Model':<20} | {'Compressed Model':<20}")
        print("-" * 70)
        
        # 打印参数
        total_b = self.baseline_metrics.get('total_params', 'N/A')
        total_c = self.compressed_metrics.get('total_params', 'N/A')
        non_zero_c = self.compressed_metrics.get('non_zero_params', 'N/A')
        sparsity_c = self.compressed_metrics.get('sparsity', 0)
        
        print(f"{'Total Parameters':<25} | {total_b:,<20} | {total_c:,<20}")
        if 'non_zero_params' in self.compressed_metrics:
            print(f"{'Non-Zero Parameters':<25} | {'N/A':<20} | {non_zero_c:,<20}")
            print(f"{'Sparsity':<25} | {'0.00%':<20} | {f'{sparsity_c:.2%}':<20}")

        # 打印FLOPs
        flops_b = self.baseline_metrics.get('flops', -1)
        flops_c = self.compressed_metrics.get('flops_theoretical', -1)
        
        print(f"{'FLOPs (M)':<25} | {f'{flops_b/1e6:.2f} M' if flops_b!=-1 else 'N/A':<20} | {f'{flops_c/1e6:.2f} M (theory)' if flops_c!=-1 else 'N/A':<20}")

        # 打印精度
        acc_b = self.baseline_metrics.get('accuracy', -1)
        acc_c = self.compressed_metrics.get('accuracy', -1)
        
        print(f"{'Accuracy':<25} | {f'{acc_b:.4f}':<20} | {f'{acc_c:.4f}':<20}")
        
        print("===========================================")