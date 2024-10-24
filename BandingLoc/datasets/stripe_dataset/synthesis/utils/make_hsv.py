import os
import shutil
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def encode_hsv_to_rgb(hsv_image):
    # 转换为NumPy数组
    hsv_array = np.array(hsv_image)
    
    # 将HSV值编码到RGB通道
    encoded_image = Image.fromarray(hsv_array, mode="RGB")
    
    return encoded_image

def process_image(input_path, output_path):
    try:
        # 打开图像并转换为HSV模式
        image = Image.open(input_path).convert("HSV")
        
        # 编码HSV值并保存到指定路径
        encoded_image = encode_hsv_to_rgb(image)
        encoded_image.save(output_path)  # 保存到新文件夹
        
        # print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def process_folder(input_folder, output_folder):
    # 创建输出图像文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图像文件
    image_files = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    # 创建处理图像任务的参数列表，包括输入和输出路径
    tasks = [(input_path, os.path.join(output_folder, os.path.basename(input_path))) for input_path in image_files]

    # 使用多进程池处理图像
    with Pool(cpu_count()) as pool:
        pool.starmap(process_image, tasks)

def copy_labels(src_folder, dst_folder):
    # 创建输出标签文件夹
    os.makedirs(dst_folder, exist_ok=True)

    # 复制所有标签文件
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        shutil.copy2(src_file, dst_file)

# 输入和输出文件夹路径
input_folder = "train/images"
output_folder = "train_hsv/images"

# 处理图像文件夹
process_folder(input_folder, output_folder)

# 复制标签文件
copy_labels("train/labels", "train_hsv/labels")
