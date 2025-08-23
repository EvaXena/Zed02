import os
from PIL import Image

def batch_convert_tiff_to_jpg(source_folder, output_folder, quality=95):
    """
    批量将指定文件夹内的所有 .tiff 和 .tif 文件转换为 .jpg 格式。

    :param source_folder: 包含 .tiff 文件的源文件夹路径。
    :param output_folder: 保存 .jpg 文件的目标文件夹路径。
    :param quality: JPG 图像的保存质量 (1-100)，默认为 95。
    """
    # 1. 检查并创建输出文件夹
    if not os.path.isdir(output_folder):
        print(f"输出文件夹 '{output_folder}' 不存在，正在创建...")
        os.makedirs(output_folder)

    # 2. 检查源文件夹是否存在
    if not os.path.isdir(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在。")
        return

    print(f"开始处理文件夹: '{source_folder}'")
    
    # 3. 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件后缀是否为 .tiff 或 .tif (忽略大小写)
        if filename.lower().endswith(('.tiff', '.tif')):
            # 构造完整的文件路径
            input_path = os.path.join(source_folder, filename)
            
            # 构造输出文件名 (将 .tiff/.tif 替换为 .jpg)
            base_name = os.path.splitext(filename)[0]
            output_filename = base_name + '.jpg'
            output_path = os.path.join(output_folder, output_filename)

            try:
                # 打开 TIFF 图像
                with Image.open(input_path) as img:
                    print(f"正在转换: {filename} ...")
                    
                    # 关键步骤：检查图像模式。JPG 不支持透明度 (alpha channel)。
                    # 如果 TIFF 是 RGBA 模式，需要转换为 RGB 模式才能保存为 JPG。
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # 保存为 JPG 格式
                    img.save(output_path, 'jpeg', quality=quality)
                    
            except Exception as e:
                print(f"转换文件 '{filename}' 时发生错误: {e}")

    print("\n所有 TIFF 文件转换完成！")
    print(f"JPG 文件已保存至: '{output_folder}'")

# --- 用户配置区域 ---
if __name__ == "__main__":
    # 请在这里修改为你的文件夹路径
    # 在 Windows 上，路径可能像这样: 'C:\\Users\\YourUser\\Pictures\\TIFF_Images'
    # 在 Linux 或 macOS 上，路径可能像这样: '/home/YourUser/Pictures/TIFF_Images'
    
    # 输入文件夹：存放 .tiff 文件的位置
    input_folder = 'flsea_dataset/imgs'
    
    # 输出文件夹：转换后的 .jpg 文件将保存在这里
    output_folder = 'flsea_dataset_jpg/imgs'
    
    # 调用函数开始转换
    batch_convert_tiff_to_jpg(input_folder, output_folder)