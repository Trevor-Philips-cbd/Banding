from PIL import Image

path = '2_.jpg'
# 打开图片
image = Image.open(path)

# 顺时针旋转90度
rotated_image = image.rotate(-90, expand=True)

# 保存旋转后的图片
rotated_image.save(path)

# 显示旋转后的图片
rotated_image.show()
