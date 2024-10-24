"""
    将整张图片切分成若干切片
"""
import os
from PIL import Image, ExifTags

def correct_image_orientation(image_path):
    """
        图片的旋转信息存储于Exif数据中，而不是实际修改图片的像素排列。PIL会根据原始像素数据保存，忽略旋转信息。
        所以需要从Exif数据中读取旋转信息来恢复旋转情况。
    """
    img = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # 如果没有EXIF数据，或者没有orientation信息，直接跳过
        pass

    return img

def split_image_vertically(image_path, output_dir, splits=5):
    # 获取图像文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 打开图像
    # img = Image.open(image_path)
    img = correct_image_orientation(image_path)
    
    # 获取图像的宽度和高度
    width, height = img.size
    
    # 计算每个等份的宽度
    slice_width = width // splits
    
    # 循环裁剪并保存图像
    for i in range(splits):
        left = i * slice_width
        right = (i + 1) * slice_width if i < splits else width  # 确保最后一份包含剩余像素
        
        # 裁剪图像
        img_slice = img.crop((left, 0, right, height))
        
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"{file_name}_part{i + 1}.png")
        
        # 保存图像
        img_slice.save(output_path)
    
    print(f"图像 {file_name} 已成功分割并保存。")

def process_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有图像文件
    for file_name in os.listdir(input_folder):
        # 构建图像的完整路径
        image_path = os.path.join(input_folder, file_name)
        
        # 确保处理的是图片文件（例如 .jpg, .png 等）
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            split_image_vertically(image_path, output_folder)

    print("文件夹中所有图像已处理完成。")

# 示例使用
input_folder = "../datasets/stripe_dataset/to_detects/to_detect"  # 输入文件夹路径
output_folder = "../datasets/stripe_dataset/to_detects/to_detect_split_5"  # 输出文件夹路径
process_folder(input_folder, output_folder)
