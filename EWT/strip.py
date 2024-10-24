import cv2
import numpy as np
import os
import random

def generate_striped_mask(image_path, stripe_width):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建与图像同样大小的mask
    mask = np.ones(gray_image.shape, dtype=np.uint8) * 255

    # 应用等距黑色条带
    for i in range(0, mask.shape[0], stripe_width * 2):
        mask[i:i + stripe_width] = 0

    # 将灰度图扩展为三通道
    gray_image_colored = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 用灰度图替代黑色部分
    masked_image = np.where(mask[:, :, np.newaxis] == 0, gray_image_colored, image)

    return mask, masked_image

def process_images(input_folder, output_folder, stripe_width):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            mask, masked_image = generate_striped_mask(image_path, stripe_width)

            # 保存mask和处理后的图像
            # cv2.imwrite(os.path.join(output_folder, f'mask_{filename}'), mask)
            cv2.imwrite(os.path.join(output_folder, f'{filename}'), masked_image)

# 示例使用
process_images('/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR', '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/X1', stripe_width=40)
# input_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR'  # 替换为你的输入文件夹路径
# output_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/X1'  # 替换为你的输出文件夹路径