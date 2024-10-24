from multiprocessing import Pool, cpu_count
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks

# 设置输入和输出文件夹路径
input_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR'
output_image_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/temp'
output_xml_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/xml_files'

# 创建输出文件夹
# os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_xml_folder, exist_ok=True)

def find_single_peak_length(signal):
    # 找到波峰的位置
    peaks, _ = find_peaks(signal) ###### f(y)和y长度一样，下标应该是一致的

    if len(peaks) == 0:
        return None  # 如果没有找到波峰，返回 None

    peak = peaks[0]  # 只处理第一个波峰

    # 向左找到峰的起始点（第一个从0到大于0的点）
    start = peak
    while start > 0 and signal[start] > 1e-2:
        start -= 1

    # 向右找到峰的终点（第一个从大于0到0的点）
    end = peak
    while end < len(signal) - 1 and signal[end] > 1e-2: 
        end += 1

    # 计算峰的长度
    peak_length = end - start

    return peaks, peak_length

def add_colored_flicker_stripes(image, stripe_freq, stripe_amplitude, direction='horizontal',
                                stripe_color=(255, 0, 0)):
    h, w = image.shape[:2]
    stripe_height = h // stripe_freq  # 每个条带的高度

    if direction == 'horizontal':
        y = np.linspace(0, 2 * np.pi * stripe_freq, h)
        sin_wave = np.power(np.sin(y), 16)
        stripe_mask = np.tile(sin_wave, (w, 1)).T
    elif direction == 'vertical': # 怎么没有power了
        x = np.linspace(0, 2 * np.pi * stripe_freq, w)
        sin_wave = np.sin(x)
        stripe_mask = np.tile(sin_wave, (h, 1))
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    
    peaks, peak_length = find_single_peak_length(sin_wave)

    stripe_mask = (stripe_mask - stripe_mask.min()) / (stripe_mask.max() - stripe_mask.min())
    stripe_mask = stripe_mask * stripe_amplitude

    stripe_b = stripe_mask * (stripe_color[0] / 255.0)
    stripe_g = stripe_mask * (stripe_color[1] / 255.0)
    stripe_r = stripe_mask * (stripe_color[2] / 255.0)

    image_with_stripes = image.astype(np.float32)
    image_with_stripes[:, :, 0] += stripe_b
    image_with_stripes[:, :, 1] += stripe_g
    image_with_stripes[:, :, 2] += stripe_r

    image_with_stripes = np.clip(image_with_stripes, 0, 255).astype(np.uint8)

    return image_with_stripes, stripe_height, peaks, peak_length


def create_xml_annotation(image_shape, stripe_color, boxes, output_xml):
    h, w = image_shape[:2]
    root = ET.Element("annotation")

    for b in boxes:
        object_elem = ET.SubElement(root, "object")
        ET.SubElement(object_elem, "name").text = "Flicker Stripe"
        ET.SubElement(object_elem, "color").text = str(stripe_color)
        ET.SubElement(object_elem, "bndbox")

        bndbox = object_elem.find("bndbox")
        ET.SubElement(bndbox, "xmin").text = str(b[0][0])
        ET.SubElement(bndbox, "ymin").text = str(b[0][1])
        ET.SubElement(bndbox, "xmax").text = str(b[1][0])
        ET.SubElement(bndbox, "ymax").text = str(b[1][1])

    tree = ET.ElementTree(root)
    tree.write(output_xml)

def xywh2xyxy(box):
    x, y, w, h = box
    return (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2))

def process_image(image_path, stripe_freq=11):
    
    # 读取输入图像
    image = cv2.imread(image_path)
    image_with_stripes, _, peaks, peak_length = add_colored_flicker_stripes(image, stripe_freq=stripe_freq, stripe_amplitude=400,
                                                                    direction='horizontal', stripe_color=(-60, 0, -2))

    H, W = image.shape[:2]

    # 获取各个框的坐标
    boxes = []
    for y in peaks:
        x = W / 2
        w = W
        h = peak_length
        boxes.append(xywh2xyxy((x, y, w, h)))

    # 在图像上可视化框
    for b in boxes:
        cv2.rectangle(image, b[0], b[1], color=(0, 255, 0), thickness=2)
    
    # 输出条带图像
    output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image_with_stripes)

    # 创建 XML 标注文件
    output_xml_path = os.path.join(output_xml_folder, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    create_xml_annotation(image.shape, (-70, 0, -70), boxes, output_xml_path)

def main():
    """多进程处理"""
    # 获取所有要处理的文件
    files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.lower().endswith(('.jpeg', 'png', 'jpg'))]

    print(cpu_count())
    with Pool(cpu_count()) as pool:
        pool.map(process_image, files)

    print("Batch processing completed.")

if __name__ == "__main__":
    main()