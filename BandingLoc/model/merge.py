"""
    将一个图片的几个切片合并
"""
import os
from PIL import Image, ImageDraw
from collections import defaultdict

def merge_images_in_folder(input_folder, output_folder, splits=5):
    # 创建一个字典用于存储每个原始图像对应的切片
    images_dict = defaultdict(list)
    
    # 遍历输入文件夹中的所有图像文件
    for file_name in os.listdir(input_folder):
        # 确保文件是图片
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 通过下划线拆分文件名，获取原始图片名（下划线前的部分）
            original_name = file_name.split('_')[0]
            # 构建完整的文件路径
            image_path = os.path.join(input_folder, file_name)
            # 将每个切片按原始图片名存储在字典中
            images_dict[original_name].append(image_path)

    # 处理每个原始图片名对应的切片
    for original_name, parts in images_dict.items():
        # 确保图像有3个切片
        if len(parts) == splits:
            # 按顺序排序切片，确保 _part1, _part2, _part3 的顺序
            parts.sort()

            # 打开所有切片
            slices = [Image.open(part) for part in parts]
            
            # 获取每个切片的宽度和高度
            widths, heights = zip(*(img.size for img in slices))
            total_width = sum(widths)
            max_height = max(heights)
            
            # 创建一个空白图像用于拼接
            new_img = Image.new('RGB', (total_width, max_height))

            # 创建ImageDraw对象以便绘制线条
            draw = ImageDraw.Draw(new_img)
            
            # 开始拼接图片
            current_width = 0
            for idx, img in enumerate(slices):
                # 粘贴切片
                new_img.paste(img, (current_width, 0))
                current_width += img.width

                # 在每两个切片之间画白线（最后一个切片后不需要）
                if idx < len(slices) - 1:
                    draw.line([(current_width, 0), (current_width, max_height)], fill="white", width=2)
                    current_width += 2  # 预留白线的宽度
            
            # 构建输出文件路径并保存拼接后的图像
            output_path = os.path.join(output_folder, f"{original_name}_merged.png")
            new_img.save(output_path)
            
            print(f"图像 {original_name} 已成功拼接为 {output_path}")
        else:
            print(f"图像 {original_name} 缺少切片，无法拼接。")

# # 示例使用
# input_folder = "runs/detect/exp62"  # 存放切片的文件夹路径
# output_folder = "runs/merged/exp62"  # 拼接后的图像存储路径

# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# merge_images_in_folder(input_folder, output_folder)

left, right = 68, 77
for i in range(left, right+1):
    input_folder = f"runs/detect/exp{i}"
    output_folder = f"runs/merged/exp{i}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    merge_images_in_folder(input_folder, output_folder, splits=3)
