import os


def generate_image_list(folder_path, output_file='image_paths.txt'):
    """
    生成包含所有jpg图像路径的txt文件
    :param folder_path: 包含jpg图像的文件夹路径
    :param output_file: 输出的txt文件名（默认：image_paths.txt）
    """
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    # 获取所有jpg文件（不区分大小写）
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]

    # 检查是否有找到jpg文件
    if not image_files:
        raise RuntimeError(f"在 {folder_path} 中没有找到jpg图像文件")

    # 生成完整路径并排序
    full_paths = [os.path.abspath(os.path.join(folder_path, f))
                  for f in sorted(image_files)]

    # 统一使用Linux风格路径（可选）
    full_paths = [p.replace('\\', '/') for p in full_paths]

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_paths))

    print(f"成功生成 {len(full_paths)} 条路径到 {output_file}")


# 使用示例
if __name__ == "__main__":
    # 需要修改为你的实际路径
    dataset_folder = r"C:\Users\Wu Meishun\Desktop\Python\YOLOV8\datasets\NEUFast\val"

    try:
        generate_image_list(dataset_folder)
    except Exception as e:
        print(f"操作失败: {str(e)}")