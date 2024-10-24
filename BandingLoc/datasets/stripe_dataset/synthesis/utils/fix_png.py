from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import cv2
from skimage import io

path = r"../trains/train/images/"

def process_image(filename):
    image = io.imread(path + filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png', image)[1].tofile(path + filename)

def main():
    fileList = os.listdir(path)

    # 使用多进程处理
    with Pool(cpu_count()) as pool:
        # 使用 tqdm 包装 pool.imap 以显示进度条
        for _ in tqdm(pool.imap_unordered(process_image, fileList), total=len(fileList)):
            pass

    print("Batch processing completed.")

if __name__ == "__main__":
    main()