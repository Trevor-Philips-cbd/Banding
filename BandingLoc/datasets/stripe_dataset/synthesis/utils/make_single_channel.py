import os
from PIL import Image
import multiprocessing

# 定义输入和输出文件夹路径
input_folder = "train/images"
output_folder_r = "train_r/images"
output_folder_g = "train_g/images"
output_folder_b = "train_b/images"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder_r, exist_ok=True)
os.makedirs(output_folder_g, exist_ok=True)
os.makedirs(output_folder_b, exist_ok=True)


def process_image(filename):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):  # 支持常见的图像格式
        # 加载图像
        img_path = os.path.join(input_folder, filename)
        image = Image.open(img_path)
        image_rgb = image.convert("RGB")

        # 拆分为三个通道
        r, g, b = image_rgb.split()

        # 保存每个通道的图像
        r_image = Image.merge("RGB", (r, Image.new('L', r.size), Image.new('L', r.size)))
        g_image = Image.merge("RGB", (Image.new('L', g.size), g, Image.new('L', g.size)))
        b_image = Image.merge("RGB", (Image.new('L', b.size), Image.new('L', b.size), b))

        # 构建保存路径并保存图像
        r_image.save(os.path.join(output_folder_r, filename))
        g_image.save(os.path.join(output_folder_g, filename))
        b_image.save(os.path.join(output_folder_b, filename))
        print(f"Processed {filename}")

if __name__ == "__main__":
    # 获取输入文件夹中所有图像文件名的列表
    filenames = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff"))]

    # 使用多进程处理
    with multiprocessing.Pool() as pool:
        pool.map(process_image, filenames)

    print("图像处理完成！")
