import os
import glob

# --- 配置 ---
# 和你的主脚本保持一致
DATA_DIR = 'decepp_train_dataset'

# --- 检查 ---
# 1. 解析成绝对路径
# 这会告诉我们 Python 认为这个文件夹的完整路径是什么
abs_data_dir = os.path.abspath(DATA_DIR)
print(f"Python 正在检查这个绝对路径: {abs_data_dir}")

# 2. 检查目录是否存在
if not os.path.isdir(abs_data_dir):
    print("错误: Python 无法找到这个目录或它不是一个目录。")
else:
    print("成功: Python 确认该路径是一个目录。")

    # 3. 使用 glob 库进行递归搜索
    # 这和 tf.data.Dataset.list_files 的行为非常相似
    print("\n正在使用 glob 搜索文件...")
    
    # 构造两种模式
    pattern_lower = os.path.join(abs_data_dir, '**/*.jpg')
    pattern_upper = os.path.join(abs_data_dir, '**/*.JPG')
    
    # recursive=True 是 ** 的关键
    files_lower = glob.glob(pattern_lower, recursive=True)
    files_upper = glob.glob(pattern_upper, recursive=True)
    
    all_files = files_lower + files_upper

    if not all_files:
        print("错误: glob 搜索失败，没有找到任何 .jpg 或 .JPG 文件。")
        print(f"使用的模式是: {pattern_lower} 和 {pattern_upper}")
    else:
        print(f"成功: glob 找到了 {len(all_files)} 个文件！")
        print("找到的前5个文件示例:")
        for f in all_files[:5]:
            print(f"  - {f}")