"""
    # imagenet100: 100个类，每个1300张图片
    # 准备6000张训练图片
    # 每个类随机采样60张
"""

import os
import shutil
import random

# 根目录
root_dir = 'imagenet100'
# 输出目录
output_dir = 'images'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历train.X1, train.X2, train.X3, train.X4
for train_dir in ['train.X1', 'train.X2', 'train.X3', 'train.X4']:
    full_train_dir = os.path.join(root_dir, train_dir)
    
    # 遍历每个类的目录
    for class_dir in os.listdir(full_train_dir):
        class_path = os.path.join(full_train_dir, class_dir)
        if os.path.isdir(class_path):
            # 获取所有jpg文件
            all_images = [img for img in os.listdir(class_path) if img.endswith('.JPEG')]
            # 随机选择60张图片
            sampled_images = random.sample(all_images, 60)
            
            # 复制图片到输出目录
            for img in sampled_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(output_dir, img)
                shutil.copyfile(src_path, dst_path)

print("图片采样完成并复制到指定目录")
