import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def encode_hsv_to_rgb(hsv_image):
    # 转换为NumPy数组
    hsv_array = np.array(hsv_image)
    
    # 将HSV值编码到RGB通道
    encoded_image = Image.fromarray(hsv_array, mode="RGB")
    
    return encoded_image

def process_image(input_path):
    try:
        # 打开图像并转换为HSV模式
        image = Image.open(input_path).convert("HSV")
        print(image.size)
        
        # 编码HSV值并保存
        encoded_image = encode_hsv_to_rgb(image)
        encoded_image.save(input_path)  # 覆盖保存到原始文件
        print(encoded_image.size)
        
        # print(f"Processed and saved: {input_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def process_folder(input_folder):
    # 获取输入文件夹中的所有图像文件
    image_files = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    # 使用多进程池处理图像
    with Pool(cpu_count()) as pool:
        pool.map(process_image, image_files)

# 输入文件夹路径
input_folder = "."

# 处理整个文件夹
process_folder(input_folder)
