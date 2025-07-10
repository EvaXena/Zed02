# comparator.py (The Perfected Version by Kurisu Makise)

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import os

# 教程里的plotting.py不是为回归任务设计的，我们这里不再需要它

class Comparator:
    def __init__(self, baseline_model_path: str, compressed_model_path: str):
        print("--- Initializing Model Comparator ---")
        self.baseline_model = self._load_model(baseline_model_path, "Baseline")
        self.compressed_model = self._load_model(compressed_model_path, "Compressed")
        
        self.baseline_metrics = {}
        self.compressed_metrics = {}
        print("--- Models loaded successfully ---\n")
        
    def _load_model(self, model_path: str, model_name: str) -> tf.keras.Model:
        print(f"Loading {model_name} model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name} model not found at: {model_path}")
        return load_model(model_path)

    def _count_non_zero_weights(self, model: tf.keras.Model) -> tuple[int, int]:
        non_zero, total = 0, 0
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                for w in layer.get_weights():
                    non_zero += tf.math.count_nonzero(w).numpy()
                    total += tf.size(w).numpy()
        return non_zero, total
    
    def _get_flops(self, model: tf.keras.Model) -> int:
        try:
            input_shape = [1] + model.input_shape[1:]
            inputs = [tf.TensorSpec(input_shape, model.inputs[0].dtype)]#生成input
            forward_pass = tf.function(model.call, input_signature=inputs)#生成静态图
            # 修正：使用tf.compat.v1.profiler.profile来获取FLOPs
            graph_info = tf.compat.v1.profiler.profile(
                graph=forward_pass.get_concrete_function().graph,
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
            return graph_info.total_float_ops // 2
        #使用静态图可能出错
        except Exception:
            return -1
        
    def compare_model_size(self):
        print("--- Round 1: Model Size (Parameter Count) ---")
        
        # 修正点一：必须同时计算并显示基准模型的参数！
        baseline_total_params = self.baseline_model.count_params()
        self.baseline_metrics['total_params'] = baseline_total_params
        print(f"Baseline Model Total Parameters:   {baseline_total_params:,}")

        # 计算压缩模型
        non_zero, total = self._count_non_zero_weights(self.compressed_model)
        self.compressed_metrics['total_params'] = total
        self.compressed_metrics['non_zero_params'] = non_zero
        sparsity = 1.0 - (non_zero / total) if total > 0 else 0
        self.compressed_metrics['sparsity'] = sparsity
        
        print(f"Compressed Model Total Parameters: {total:,}")
        print(f"Compressed Model Non-Zero Params:  {non_zero:,}")
        print(f"Achieved Sparsity:                 {sparsity:.2%}")
        print("--------------------------------------------\n")
        return self

    def compare_flops(self):
        print("--- Round 2: Computational Cost (FLOPs) ---")
        # ... (这部分你的代码是正确的，无需修改) ...
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
    
    def compare_regression_performance(self, test_dataset: tf.data.Dataset, batch_size: int = None):

        print("--- Round 3: Predictive Performance (Regression) ---")
        
        print("Evaluating baseline model...")
        results_baseline = self.baseline_model.evaluate(test_dataset, verbose=0, return_dict=True)
        
        print("Evaluating compressed model...")
        results_compressed = self.compressed_model.evaluate(test_dataset, verbose=0, return_dict=True)
        
        self.baseline_metrics.update(results_baseline)
        self.compressed_metrics.update(results_compressed)
        
        print(f"\n{'Metric':<25} | {'Baseline Model':<20} | {'Compressed Model':<20}")
        print("-" * 70)
        # 使用.get()方法，以防某个模型没有编译某个metric
        print(f"{'Loss (MSE)':<25} | {results_baseline.get('loss', 'N/A'):<20.4f} | {results_compressed.get('loss', 'N/A'):<20.4f}")
        print(f"{'Mean Absolute Error':<25} | {results_baseline.get('mean_absolute_error', 'N/A'):<20.4f} | {results_compressed.get('mean_absolute_error', 'N/A'):<20.4f}")
        
        print("\nNote: Lower values are better for both metrics.")
        print("------------------------------------------------------\n")
        return self
        
    def generate_summary_report(self):
        """修正点三：现在可以正确报告所有指标的最终总结报告。"""
        print("========== FINAL COMPARISON REPORT ==========")
        print(f"{'Metric':<25} | {'Baseline Model':<20} | {'Compressed Model':<20}")
        print("-" * 70)
        
        # 参数对比
        total_b = self.baseline_metrics.get('total_params', 'N/A')
        total_c = self.compressed_metrics.get('total_params', 'N/A')
        non_zero_c = self.compressed_metrics.get('non_zero_params', 'N/A')
        
        # --- 关键修正！解除嵌套！ ---
        # 1. 先把内层的计算，赋值给一个变量
        sparsity_val = self.compressed_metrics.get("sparsity", 0)
        sparsity_str = f"{sparsity_val:.2%}" 
        
        print(f"{'Total Parameters':<25} | {total_b:,<20} | {total_c:,<20}")
        if 'non_zero_params' in self.compressed_metrics:
            print(f"{'Non-Zero Parameters':<25} | {'N/A':<20} | {non_zero_c:,<20}")
            # 2. 然后，把这个已经计算好的字符串变量，放进外层的f-string里
            print(f"{'Sparsity':<25} | {'0.00%':<20} | {sparsity_str:<20}")
        # --- 修正结束 ---

        # FLOPs对比
        flops_b = self.baseline_metrics.get('flops', -1)
        flops_c = self.compressed_metrics.get('flops_theoretical', -1)
        print(f"{'FLOPs (M-theory)':<25} | {f'{flops_b/1e6:.2f} M' if flops_b!=-1 else 'N/A':<20} | {f'{flops_c/1e6:.2f} M' if flops_c!=-1 else 'N/A':<20}")

        # 性能指标对比
        loss_b = self.baseline_metrics.get('loss', 'N/A')
        loss_c = self.compressed_metrics.get('loss', 'N/A')
        mae_b = self.baseline_metrics.get('mean_absolute_error', 'N/A')
        mae_c = self.compressed_metrics.get('mean_absolute_error', 'N/A')

        # 修正！你之前打印的是accuracy，但metrics里存的是mae和loss
        print(f"{'Loss (MSE)':<25} | {loss_b:<20.4f} | {loss_c:<20.4f}")
        print(f"{'Mean Absolute Error':<25} | {mae_b:<20.4f} | {mae_c:<20.4f}")
        
        print("===========================================")