import os
from PIL import Image

# 设置图像文件夹路径
input_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR_beifen'
output_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 批量转换图像
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 可以添加更多格式
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('L')  # 转换为灰度图
        img.save(os.path.join(output_folder, filename))

print("转换完成！")
