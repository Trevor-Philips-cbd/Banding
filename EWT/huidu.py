import cv2
import xml.etree.ElementTree as ET
import os
import glob

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def replace_bboxes_with_gray(image, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for bbox in root.findall('.//bndbox'):
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        roi = image[ymin:ymax, xmin:xmax]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi_colored = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

        image[ymin:ymax, xmin:xmax] = gray_roi_colored

    return image

def batch_process_images(image_dir, xml_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像和XML文件
    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    xml_paths = [os.path.splitext(os.path.basename(path))[0] + '.xml' for path in image_paths]
    xml_paths = [os.path.join(xml_dir, xml) for xml in xml_paths]

    # 遍历图像和XML文件
    for image_path, xml_path in zip(image_paths, xml_paths):
        # 读取图像
        image = cv2.imread(image_path)

        # 替换标注框
        processed_image = replace_bboxes_with_gray(image, xml_path)

        # 保存输出图像
        # print(image_path[:-4])
        # output_path = os.path.join(output_dir, os.path.basename(image_path[:-4] + '_g'+'.png'))
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        print(output_path)
        cv2.imwrite(output_path, processed_image)

# 示例调用
batch_process_images('/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/temp',
                     '/mnt/sdc/org/home/bdc/EWT/datasets/xml_files',
                     '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/X1')





# from PIL import Image
# import os
# import numpy as np
#
# # 设置包含图像的文件夹路径
# input_folder = '/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_LR_bicubic/X1'
# # 设置转换后的图像保存的文件夹路径
# output_folder = '/mnt/sdc/org/home/bdc/EWT/temp/h'
#
# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 遍历输入文件夹中的所有文件
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
#         # 构建完整的文件路径
#         img_path = os.path.join(input_folder, filename)
#         # 打开图像
#         with Image.open(img_path) as img:
#             # 转换为灰度图像
#             gray_array = np.array(img)
#             # gray_img = img.convert('L')
#             # gray_array = np.array(gray_img)
#             # if gray_array.shape[2]
#             if gray_array.shape[1] == 0:
#                 print(img_path)
#                 print(gray_array.shape)
#             # 构建输出文件的完整路径
#             gray_img_path = os.path.join(output_folder, filename)
#             # 保存灰度图像
#             # gray_img.save(gray_img_path)
#             # print(f'图像 {filename} 已转换为灰度图并保存。')
