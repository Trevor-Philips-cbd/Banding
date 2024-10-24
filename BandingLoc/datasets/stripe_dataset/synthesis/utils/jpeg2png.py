"""
    将images文件夹中的jpeg图像转换为png图像，输出到png_images文件夹
"""


import os
from PIL import Image
from multiprocessing import Pool, cpu_count

def convert_image(file_info):
    input_folder, output_folder, filename = file_info
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # 打开JPEG图像
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # 构建输出文件路径
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, base_filename + '.png')

        # 保存为PNG格式
        img.save(output_path, 'PNG')
        # print(f"Converted {filename} to PNG")

def main():
    # 输入和输出文件夹路径
    input_folder = 'images'
    output_folder = 'png_images'

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有要处理的文件
    files = os.listdir(input_folder)
    file_info_list = [(input_folder, output_folder, filename) for filename in files]

    print(cpu_count())
    # 使用多进程处理
    with Pool(cpu_count()) as pool:
        pool.map(convert_image, file_info_list)

    print("Conversion completed!")

if __name__ == "__main__":
    main()
