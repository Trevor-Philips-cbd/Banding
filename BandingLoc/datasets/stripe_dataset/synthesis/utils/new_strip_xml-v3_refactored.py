"""
    * 在png图像上添加条带，输出条带图像、画框图像和xml标注文件
        - 输入：png_images
        - 输出：striped_images、boxed_images、xml_files
        
    * 重构代码
    * 修改条带宽度范围为按比例
"""

from multiprocessing import Pool, cpu_count
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

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
# stripe_width_min = 20
# stripe_width_max = 25
stripe_opacity_min = 128  # 透明度范围是0（完全透明）到255（完全不透明）
stripe_opacity_max = 180

# 定义条带颜色（橙色）
stripe_color = (255, 165, 100)  # 橙色（R=255, G=165, B=0）


def create_annotation_element(image_path, width, height):
    """创建一个XML格式的标注元素"""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "4"  # RGBA
    ET.SubElement(annotation, "segmented").text = "0"
    return annotation


def add_stripe_to_annotation(annotation, x, stripe_width, height):
    """在给定的XML元素对象中添加一个新的子对象来表示条纹"""
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = "stripe"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x)
    ET.SubElement(bndbox, "ymin").text = "0"
    ET.SubElement(bndbox, "xmax").text = str(x + stripe_width)
    ET.SubElement(bndbox, "ymax").text = str(height)


def draw_fade_stripe(draw, start_x, stripe_width, width, height, stripe_opacity, mask):
    """画一张图像上的条带"""
    if start_x + stripe_width > width:
        return

    def process_color(x):
        """在宽度x处画竖直条带，深色区域和浅色区域使用不同的不透明度"""
        x_mask = mask[:, x]
        current_color = gradient_color if x_mask[0] == 0 else gradient_color_dark
        start_y = 0

        for y in range(1, height):
            if x_mask[y] != x_mask[y - 1]:
                draw.line([(x, start_y), (x, y - 1)], fill=current_color, width=1)
                start_y = y
                current_color = gradient_color if x_mask[y] == 0 else gradient_color_dark
        draw.line([(x, start_y), (x, height - 1)], fill=current_color, width=1)

    for i in range(stripe_width // 2):
        opacity = int(stripe_opacity * (i / (stripe_width // 2))) if i < stripe_width * 0.35 else int(
            stripe_opacity * (stripe_width * 0.35) / (stripe_width // 2))
        gradient_color = (stripe_color[0], stripe_color[1], stripe_color[2], opacity)
        gradient_color_dark = (stripe_color[0], stripe_color[1], stripe_color[2], opacity // 3)
        process_color(start_x + i)
        process_color(start_x + stripe_width - 1 - i)

    for i in range(stripe_width // 2, stripe_width - stripe_width // 2):
        process_color(start_x + i)


def process_image(image_path):
    """处理单张图像"""
    original_image = Image.open(image_path)
    striped_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    boxed_image = original_image.convert('RGBA').copy()
    draw = ImageDraw.Draw(striped_image)
    box_draw = ImageDraw.Draw(boxed_image)

    hsv_image = original_image.convert("HSV")
    _, _, v = hsv_image.split()
    mask = np.array(Image.eval(v, lambda x: 255 if 0 <= x <= 170 else 0))

    annotation = create_annotation_element(image_path, original_image.width, original_image.height)

    x = 0
    stripe_width_min = int(original_image.width * 0.05)
    stripe_width_max = int(original_image.width * 0.10)
    while x < original_image.width:
        stripe_width = random.randint(stripe_width_min, stripe_width_max)
        stripe_opacity = random.randint(stripe_opacity_min, stripe_opacity_max)
        draw_fade_stripe(draw, x, stripe_width, original_image.width, original_image.height, stripe_opacity, mask)
        add_stripe_to_annotation(annotation, x, stripe_width, original_image.height)
        box_draw.rectangle((x, 0, x + stripe_width, original_image.height), outline="red", width=2)
        x += stripe_width * 2

    final_image = Image.alpha_composite(original_image.convert('RGBA'), striped_image)
    output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
    final_image.save(output_image_path)

    boxed_image_path = os.path.join(output_boxed_image_folder, os.path.basename(image_path))
    boxed_image.save(boxed_image_path)

    output_xml_path = os.path.join(output_xml_folder, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)


# process_image(os.path.join(input_folder, "n01440764_188.png"))

def main():
    """多进程处理"""
    # 获取所有要处理的文件
    files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.lower().endswith('.png')]

    print(cpu_count())
    with Pool(cpu_count()) as pool:
        pool.map(process_image, files)

    print("Batch processing completed.")

if __name__ == "__main__":
    main()