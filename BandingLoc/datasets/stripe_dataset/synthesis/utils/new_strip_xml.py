"""
    + 两边部分渐变不透明度，向中间递增；中间部分固定不透明度
"""

from multiprocessing import Pool, cpu_count
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random

# 设置输入和输出文件夹路径
input_folder = 'png_images'
output_image_folder = 'striped_images'
output_boxed_image_folder = 'boxed_images'
output_xml_folder = 'xml_files'

# 创建输出文件夹
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_boxed_image_folder, exist_ok=True)
os.makedirs(output_xml_folder, exist_ok=True)

# 设置条带宽度和透明度的上下限
stripe_width_min = 20
stripe_width_max = 30
stripe_opacity_min = 128  # 透明度范围是0（完全透明）到255（完全不透明）
stripe_opacity_max = 180

# 定义条带颜色（黑色）
# stripe_color = (0, 0, 0)  # 黑色（R=0, G=0, B=0）

stripe_color = (255, 165, 100)  # 橙色（R=255, G=165, B=0）
# stripe_color = (255, 255, 0)  # 黄色（R=255, G=255, B=0）

def draw_fade_stripe(draw, start_x, width, height, stripe_opacity):
    for i in range(width // 2):
        if i < width*0.35:
            opacity = int(stripe_opacity * (i / (width // 2)))  # 两边部分渐变不透明度，向中间递增
        else:
            opacity = int(stripe_opacity * (width*0.35) / (width // 2)) # 中间部分固定不透明度

        gradient_color = (stripe_color[0], stripe_color[1], stripe_color[2], opacity)
        draw.line([(start_x + i, 0), (start_x + i, height)], fill=gradient_color)
        draw.line([(start_x + width - 1 - i, 0), (start_x + width - 1 - i, height)], fill=gradient_color)

    # 绘制中间部分
    for i in range(width // 2, width - width // 2):
        draw.line([(start_x + i, 0), (start_x + i, height)],
                  fill=(stripe_color[0], stripe_color[1], stripe_color[2], opacity))


def process_image(image_path):
    original_image = Image.open(image_path)

    # 创建一个和原始图像大小一样的新图像，用于绘制条带
    striped_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    boxed_image = original_image.convert('RGBA').copy()  # 用于绘制边界框

    # 创建XML文件结构
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(original_image.width)
    ET.SubElement(size, "height").text = str(original_image.height)
    ET.SubElement(size, "depth").text = "4"  # RGBA

    ET.SubElement(annotation, "segmented").text = "0"

    # 绘制半透明条带并记录坐标
    draw = ImageDraw.Draw(striped_image)
    box_draw = ImageDraw.Draw(boxed_image)
    x = 0

    while x < original_image.width:
        # 随机生成条带的宽度和透明度
        stripe_width = random.randint(stripe_width_min, stripe_width_max)
        stripe_opacity = random.randint(stripe_opacity_min, stripe_opacity_max)
        
        draw_fade_stripe(draw, x, stripe_width, original_image.height, stripe_opacity)

        # 添加条带的左上角和右下角坐标到XML文件
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "stripe"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x)
        ET.SubElement(bndbox, "ymin").text = "0"
        ET.SubElement(bndbox, "xmax").text = str(x + stripe_width)
        ET.SubElement(bndbox, "ymax").text = str(original_image.height)

        # 绘制边界框
        box_draw.rectangle([x, 0, x + stripe_width, original_image.height], outline="red", width=2)

        x += stripe_width * 2  # 为下一个条带设置起始位置

    # 合并原始图像和条带图像
    final_image = Image.alpha_composite(original_image.convert('RGBA'), striped_image)

    # 保存结果图像
    output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
    final_image.save(output_image_path)

    # 保存包含边界框的图像
    boxed_image_path = os.path.join(output_boxed_image_folder, os.path.basename(image_path))
    boxed_image.save(boxed_image_path)

    # 保存XML文件
    output_xml_path = os.path.join(output_xml_folder, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)

    # print(f"Processed {image_path}")


# # 批量处理文件夹中的所有图像
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.png')):
#         process_image(os.path.join(input_folder, filename))

# print("Batch processing completed.")

def main():
    # 获取所有要处理的文件
    files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.lower().endswith('.png')]

    print(cpu_count())
    # 使用多进程处理
    with Pool(cpu_count()) as pool:
        pool.map(process_image, files)

    print("Batch processing completed.")

if __name__ == "__main__":
    main()