import os
import shutil

def duplicate_and_rename_images(source_folder):
    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        return "源文件夹不存在"

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 只处理图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 构建原始文件和复制文件的完整路径
            original_file = os.path.join(source_folder, filename)
            # 修改文件名
            new_filename = filename[:-4] + '_g' + '.png'
            new_file = os.path.join(source_folder, new_filename)

            # 复制并重命名文件
            shutil.copy2(original_file, new_file)

    return "图像复制和重命名完成"

print(duplicate_and_rename_images("/mnt/sdc/org/home/bdc/EWT/datasets/DIV2K/DIV2K_train_HR"))